"""
AnndataBackend: shared AnnData lifecycle manager used by sc_graph_mcp_server.py.

Based on the scmcp-shared AnnData backend pattern.
"""

from typing import Optional
from pathlib import Path

import anndata as ad


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
