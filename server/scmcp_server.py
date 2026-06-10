"""
scMCP Server Implementation for Google ADK Integration

This module creates an MCP server that exposes scanpy single-cell analysis tools
to Google ADK agents while properly managing AnnData lifecycle.

Based on the scmcp-shared AnnData backend pattern.
"""

import asyncio
import nest_asyncio
from typing import Optional, List, Any
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path

from fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
# from abcoder.backend import NotebookManager  # unused import
import anndata as ad

# Import FastRAG for knowledge base search
from fast_rag import get_rag_instance


# Enable nested event loops to prevent conflicts with Google ADK
nest_asyncio.apply()


class AdataInfo(BaseModel):
    """Schema for tracking AnnData objects across requests"""
    sampleid: str = Field(description="adata sampleid")
    adtype: str = Field(default="exp", description="The input adata.X data type")

    model_config = ConfigDict(extra="ignore")


class AnndataBackend:
    """
    Backend for managing AnnData objects with persistent lifecycle.

    Supports shared data directory pattern:
    - Multiple MCP servers can share the same data_dir
    - Lazy loading: automatically loads from disk if not in memory
    - Data persistence: modifications are saved back to shared location

    This enables natural multi-server workflows where data flows between
    different analysis tools (scanpy → decoupler → cellrank → etc.)
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.active_id: Optional[str] = None
        self.adata_registry: dict[str, ad.AnnData] = {}

    def load_adata(self, sampleid: str, h5ad_path: Optional[str] = None) -> ad.AnnData:
        """
        Load AnnData from file and save to shared location.

        Args:
            sampleid: Unique identifier for this dataset
            h5ad_path: Path to h5ad file. If None, loads from data_dir/sampleid.h5ad

        Returns:
            The loaded AnnData object
        """
        # If already in memory, return it
        if sampleid in self.adata_registry:
            return self.adata_registry[sampleid]

        # Determine source path
        if h5ad_path is None:
            h5ad_path = str(self.data_dir / f"{sampleid}.h5ad")

        # Load from source
        adata = ad.read_h5ad(h5ad_path)
        adata.uns["scmcp_sampleid"] = sampleid

        # Save to shared location (in case source was elsewhere)
        shared_path = self.data_dir / f"{sampleid}.h5ad"
        if Path(h5ad_path) != shared_path:
            adata.write_h5ad(shared_path)

        # Register in memory
        self.adata_registry[sampleid] = adata
        self.active_id = sampleid
        return adata

    def get_adata(self, sampleid: Optional[str] = None) -> ad.AnnData:
        """
        Get AnnData from registry with lazy loading from shared disk.

        This is the key method for shared data directory pattern:
        - If data is in memory: return it
        - If not in memory but exists on disk: load it automatically
        - This allows different servers to access data loaded by other servers

        Args:
            sampleid: Sample to retrieve. If None, uses active_id

        Returns:
            The AnnData object

        Raises:
            ValueError: If no sampleid specified and no active_id
            FileNotFoundError: If sample doesn't exist in memory or on disk
        """
        target_id = sampleid or self.active_id
        if target_id is None:
            raise ValueError("No active sampleid. Load an AnnData first.")

        # If in memory, return it
        if target_id in self.adata_registry:
            return self.adata_registry[target_id]

        # Try to load from shared disk (lazy loading)
        shared_path = self.data_dir / f"{target_id}.h5ad"
        if shared_path.exists():
            print(f"📂 Lazy loading '{target_id}' from shared disk: {shared_path}")
            adata = ad.read_h5ad(shared_path)
            self.adata_registry[target_id] = adata
            self.active_id = target_id
            return adata

        # Not found anywhere
        raise FileNotFoundError(
            f"Sample '{target_id}' not found in memory or disk.\n"
            f"Available in memory: {list(self.adata_registry.keys())}\n"
            f"Searched path: {shared_path}"
        )

    def save_adata(self, sampleid: Optional[str] = None):
        """
        Save AnnData back to shared disk location.

        This makes modifications visible to other servers.

        Args:
            sampleid: Sample to save. If None, uses active_id

        Returns:
            Path where the file was saved
        """
        adata = self.get_adata(sampleid)

        # Get sampleid from uns or use parameter/active_id
        if 'scmcp_sampleid' in adata.uns:
            save_id = adata.uns['scmcp_sampleid']
        else:
            save_id = sampleid or self.active_id
            # Set it for future saves
            adata.uns['scmcp_sampleid'] = save_id

        save_path = self.data_dir / f"{save_id}.h5ad"
        adata.write_h5ad(save_path)
        return str(save_path)


class ScMCPServer:
    """
    MCP Server for single-cell analysis following scmcp-shared patterns.

    Key features:
    - Lifespan context manager to maintain AnnData across requests
    - nest_asyncio for event loop compatibility
    - Proper tool registration with FastMCP
    """

    def __init__(
        self,
        name: str = "scmcp-scanpy",
        instructions: Optional[str] = None,
        data_dir: str = "./data",
        enable_rag: bool = True,
    ):
        # Initialize backend for AnnData management
        self.backend = AnndataBackend(data_dir=data_dir)

        # Initialize RAG (knowledge base search)
        self.rag = None
        if enable_rag:
            try:
                print("🔍 Initializing RAG knowledge base...")
                self.rag = get_rag_instance()
                print("✓ RAG ready for knowledge search")
            except Exception as e:
                print(f"⚠️  RAG initialization failed: {e}")
                print("   Continuing without knowledge base search...")

        # Create FastMCP server with lifespan
        self.mcp = FastMCP(
            name=name,
            instructions=instructions or self._default_instructions(),
            lifespan=self._lifespan_context,
        )

        # Register tools
        self._register_tools()

    @staticmethod
    def _default_instructions() -> str:
        return """
        This is a single-cell RNA-seq analysis server providing:
        1. Scanpy analysis tools for data processing
        2. Knowledge base search for answering questions

        Key concepts:
        - sampleid: Unique identifier for each dataset
        - adtype: Data type in adata.X (default 'exp' for expression)

        When user asks questions (what/how/why):
        - Use search_knowledge tool to find relevant information
        - Provide accurate answers based on documentation

        When user requests analysis:
        - Use appropriate scanpy tools
        - Always specify sampleid to operate on the correct dataset
        """

    @asynccontextmanager
    async def _lifespan_context(self, server: FastMCP) -> AsyncIterator[Any]:
        """
        Lifespan context manager that provides backend to all tool calls.

        This is the KEY to maintaining AnnData state across requests.
        Based on: scmcp_shared/mcp_base.py:141-143
        """
        yield self.backend

    def _register_tools(self):
        """Register all MCP tools"""

        # RAG Knowledge Search Tool
        if self.rag is not None:
            @self.mcp.tool()
            async def search_knowledge(
                query: str,
                top_k: int = 3
            ) -> str:
                """
                Search the scRNA-seq knowledge base for information.

                Use this tool when:
                - User asks conceptual questions ("What is...", "How to...", "Why...")
                - Need to explain methods or best practices
                - Uncertain about parameter meanings or recommendations
                - Want to provide context or background information

                Examples:
                - "What is Leiden clustering?" → Use this tool
                - "How should I choose normalization parameters?" → Use this tool
                - "What's the difference between PCA and UMAP?" → Use this tool
                - "Load data from file.h5ad" → Don't use this tool, use load_data instead

                Args:
                    query: Question or search query
                    top_k: Number of relevant documents to return (default: 3)

                Returns:
                    Relevant documentation and explanations from knowledge base
                """
                try:
                    # Fast semantic search (< 100ms)
                    formatted_results = self.rag.search_formatted(
                        query=query,
                        top_k=top_k,
                        include_metadata=True
                    )

                    return f"""# Knowledge Base Results for: "{query}"

{formatted_results}

---
💡 These results are from the scRNA-seq documentation knowledge base.
"""
                except Exception as e:
                    return f"Error searching knowledge base: {str(e)}"

        @self.mcp.tool()
        async def load_data(
            sampleid: str,
            h5ad_path: str,
        ) -> str:
            """
            Load an AnnData object from h5ad file.

            Args:
                sampleid: Unique identifier for this dataset
                h5ad_path: Path to .h5ad file

            Returns:
                Success message with dataset info
            """
            from fastmcp.server.dependencies import get_context

            ctx = get_context()
            backend: AnndataBackend = ctx.request_context.lifespan_context

            adata = backend.load_adata(sampleid, h5ad_path)

            return f"""Loaded dataset '{sampleid}':
- Shape: {adata.shape[0]} cells × {adata.shape[1]} genes
- Active sampleid: {backend.active_id}
"""

        @self.mcp.tool()
        async def get_data_info(
            adinfo: AdataInfo,
        ) -> str:
            """
            Get information about loaded AnnData.

            Args:
                adinfo: AdataInfo with sampleid and adtype

            Returns:
                Dataset information
            """
            from fastmcp.server.dependencies import get_context

            ctx = get_context()
            backend: AnndataBackend = ctx.request_context.lifespan_context

            adata = backend.get_adata(adinfo.sampleid)

            info = f"""Dataset Info:
- Sample ID: {adata.uns.get('scmcp_sampleid')}
- Shape: {adata.shape[0]} cells × {adata.shape[1]} genes
- obs columns: {', '.join(adata.obs.columns[:10])}
- var columns: {', '.join(adata.var.columns[:10])}
- obsm keys: {', '.join(adata.obsm.keys())}
- uns keys: {', '.join(adata.uns.keys())}
"""
            return info

        @self.mcp.tool()
        async def scanpy_preprocess(
            adinfo: AdataInfo,
            min_genes: int = 200,
            min_cells: int = 3,
            n_top_genes: int = 2000,
        ) -> str:
            """
            Run standard scanpy preprocessing pipeline.

            Args:
                adinfo: AdataInfo with sampleid and adtype
                min_genes: Minimum genes per cell
                min_cells: Minimum cells per gene
                n_top_genes: Number of highly variable genes

            Returns:
                Preprocessing summary
            """
            from fastmcp.server.dependencies import get_context
            import scanpy as sc

            ctx = get_context()
            backend: AnndataBackend = ctx.request_context.lifespan_context

            adata = backend.get_adata(adinfo.sampleid)

            # Quality control
            sc.pp.filter_cells(adata, min_genes=min_genes)
            sc.pp.filter_genes(adata, min_cells=min_cells)

            # Normalization
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            # Feature selection
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

            # Log operation
            if "operation" not in adata.uns:
                adata.uns["operation"] = {"op": {}, "opid": []}

            return f"""Preprocessing completed:
- Filtered to {adata.shape[0]} cells × {adata.shape[1]} genes
- Selected {adata.var['highly_variable'].sum()} highly variable genes
"""

        @self.mcp.tool()
        async def scanpy_pca(
            adinfo: AdataInfo,
            n_comps: int = 50,
            use_highly_variable: bool = True,
        ) -> str:
            """
            Compute PCA.

            Args:
                adinfo: AdataInfo with sampleid and adtype
                n_comps: Number of principal components
                use_highly_variable: Use only highly variable genes

            Returns:
                PCA computation summary
            """
            from fastmcp.server.dependencies import get_context
            import scanpy as sc

            ctx = get_context()
            backend: AnndataBackend = ctx.request_context.lifespan_context

            adata = backend.get_adata(adinfo.sampleid)

            sc.pp.pca(adata, n_comps=n_comps, use_highly_variable=use_highly_variable)

            variance_ratio = adata.uns['pca']['variance_ratio'][:10]

            return f"""PCA completed:
- Computed {n_comps} principal components
- Top 10 variance ratios: {', '.join([f'{v:.3f}' for v in variance_ratio])}
- PCA stored in adata.obsm['X_pca']
"""

        @self.mcp.tool()
        async def scanpy_neighbors(
            adinfo: AdataInfo,
            n_neighbors: int = 15,
            n_pcs: int = 40,
        ) -> str:
            """
            Compute neighborhood graph.

            Args:
                adinfo: AdataInfo with sampleid and adtype
                n_neighbors: Number of neighbors
                n_pcs: Number of PCs to use

            Returns:
                Neighbor graph summary
            """
            from fastmcp.server.dependencies import get_context
            import scanpy as sc

            ctx = get_context()
            backend: AnndataBackend = ctx.request_context.lifespan_context

            adata = backend.get_adata(adinfo.sampleid)

            sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

            return f"""Computed k-nearest neighbor graph:
- n_neighbors: {n_neighbors}
- n_pcs: {n_pcs}
- Graph stored in adata.obsp['connectivities']
"""

        @self.mcp.tool()
        async def scanpy_umap(
            adinfo: AdataInfo,
            min_dist: float = 0.5,
        ) -> str:
            """
            Compute UMAP embedding.

            Args:
                adinfo: AdataInfo with sampleid and adtype
                min_dist: Minimum distance parameter

            Returns:
                UMAP computation summary
            """
            from fastmcp.server.dependencies import get_context
            import scanpy as sc

            ctx = get_context()
            backend: AnndataBackend = ctx.request_context.lifespan_context

            adata = backend.get_adata(adinfo.sampleid)

            sc.tl.umap(adata, min_dist=min_dist)

            return f"""UMAP completed:
- min_dist: {min_dist}
- Embedding stored in adata.obsm['X_umap']
- Shape: {adata.obsm['X_umap'].shape}
"""

        @self.mcp.tool()
        async def scanpy_leiden(
            adinfo: AdataInfo,
            resolution: float = 1.0,
        ) -> str:
            """
            Perform Leiden clustering.

            Args:
                adinfo: AdataInfo with sampleid and adtype
                resolution: Resolution parameter

            Returns:
                Clustering summary
            """
            from fastmcp.server.dependencies import get_context
            import scanpy as sc

            ctx = get_context()
            backend: AnndataBackend = ctx.request_context.lifespan_context

            adata = backend.get_adata(adinfo.sampleid)

            sc.tl.leiden(adata, resolution=resolution)

            n_clusters = adata.obs['leiden'].nunique()
            cluster_sizes = adata.obs['leiden'].value_counts().to_dict()

            return f"""Leiden clustering completed:
- Resolution: {resolution}
- Number of clusters: {n_clusters}
- Cluster sizes: {cluster_sizes}
- Results stored in adata.obs['leiden']
"""

        @self.mcp.tool()
        async def save_data(
            adinfo: AdataInfo,
        ) -> str:
            """
            Save AnnData to disk.

            Args:
                adinfo: AdataInfo with sampleid

            Returns:
                Save location
            """
            from fastmcp.server.dependencies import get_context

            ctx = get_context()
            backend: AnndataBackend = ctx.request_context.lifespan_context

            save_path = backend.save_adata(adinfo.sampleid)

            return f"Saved to: {save_path}"


def main():
    """Entry point for running the MCP server via streamable-http"""
    import os

    # Set environment variables for scMCP
    os.environ["SCMCP_TRANSPORT"] = "streamable-http"
    os.environ["SCMCP_LOG_FILE"] = "./scmcp.log"

    # Create and run server
    server = ScMCPServer(
        name="scmcp-scanpy",
        data_dir="./data"
    )

    # Run with streamable-http transport (MCP latest standard)
    print("🚀 Starting scMCP server with streamable-http on port 8000...")
    server.mcp.run(transport="streamable-http", port=8000)


if __name__ == "__main__":
    main()
