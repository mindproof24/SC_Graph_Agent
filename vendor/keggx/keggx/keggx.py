#!/usr/bin/env python3

"""Modified KEGG KGML parser used by SC_Graph_Agent.

Derived from keggx v0.1.0 by Johnny Li:
https://github.com/iamjli/keggx
Upstream commit: 10b30ca7c70587b6b1387211badd5bc9aebfbe09

The vendored implementation modifies KGML relation parsing, official gene-symbol
mapping, group-node expansion, package-resource handling, and integration with
the SC_Graph_Agent server workflow. See the LICENSE and NOTICE files in the
vendored package root.
"""


import numpy as np
import pandas as pd
import networkx as nx

import xml.etree.ElementTree as ET

from itertools import combinations, product

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import seaborn as sns

from keggx.draw import Node, set_grid, shortest_arrow
#from .draw import Node, set_grid, shortest_arrow


class KEGG:

    def __init__(self, pathway_id=None, KGML_file=None, compound=False, replace_with_official_gene = True):
        if compound:
            raise NotImplementedError("compound=True is not supported in the vendored SC Graph Agent keggx parser.")

        # Set pathway metadata attributes
        if pathway_id is not None: 
            import requests
            self.root = ET.fromstring(requests.get('http://rest.kegg.jp/get/{}/kgml'.format(pathway_id)).text)
        elif KGML_file is not None:
            self.root = ET.parse(KGML_file).getroot()
        else: 
            print('Need to specify `pathway_id` or `KGML_file`.')
        
        
        self.name   = self.root.get('name') 
        self.org    = self.root.get('org') 
        self.number = self.root.get('number')
        self.title  = self.root.get('title') 
        self.link   = self.root.get('link')
        
        self.compound = compound
        self.replace_with_official_gene = replace_with_official_gene
        
        # Each of the 3 types of elements allowed in KGML files
        self._entries   = self.root.findall('entry')
        self._reactions = self.root.findall('reaction')
        self._relations = self.root.findall('relation')
        self._groups    = self.root.findall('.//entry[@type="group"]')

        # DataFrame columns
        self.node_columns = ['id', 'name', 'aliases', 'type', 'x', 'y', 'height', 'width', 'shape', 'bgcolor', 'fgcolor']
        self.edge_columns = ['source', 'target', 'effect', 'indirect', 'modification', 'type']

        # Graph attribute DataFrames
        self.entry_attributes_df = self._get_entry_attributes_as_dataframe()
        self.node_attributes_df  = self._get_node_attributes_as_dataframe()
        self.edge_attributes_df  = self._get_edge_attributes_as_dataframe()

        #self.inferred_edge_attributes_df = self._infer_gene_edges_from_reactions()

    #### NODES ####

    def _get_entry_attributes_as_dataframe(self):
        """
        Parse entry elements and store attributes in a dataframe. Entries include all KEGG pathway 
        elements, incuding maplinks, orthologs, etc.

        Returns: 
            pandas.DataFrame: entry attributes
        """

        entry_type_df      = pd.DataFrame([entry.attrib for entry in self._entries]).drop(columns=['name', 'link'], errors='ignore')
        entry_graphics_df = pd.DataFrame([entry.find('graphics').attrib for entry in self._entries]).rename(columns={'name': 'aliases', 'type': 'shape'})

        entry_attributes_df = pd.concat([entry_type_df, entry_graphics_df], axis=1)
        entry_attributes_df['aliases'] = entry_attributes_df['aliases'].fillna('')
        entry_attributes_df['name'] = entry_attributes_df['aliases'].apply(lambda x: x.split(', ')[0].rstrip('.'))
        entry_attributes_df = entry_attributes_df[self.node_columns].set_index('id')


        entry_attributes_df[['x', 'y', 'height', 'width']] = entry_attributes_df[['x', 'y', 'height', 'width']].astype(float)

        return entry_attributes_df


    def _get_node_attributes_as_dataframe(self, types=['gene', 'compound']): 
        """
        Gets entry elements of a particular type.

        Arguments:
            types (list): element types to keep

        Returns: 
            pandas.DataFrame
        """

        return self.entry_attributes_df[self.entry_attributes_df['type'].isin(types)]


    #### EDGES ####

    def _get_edge_attributes_as_dataframe(self,replace_with_official_gene = True): 
        """
        Logic for converting reaction and relation elements into a dataframe of edge attributes

        Returns: 
            pandas.DataFrame: edge attributes 
        """
        
        #We don't consider compound information for scRNA-seq analysis.
        if self.compound==False:
            edge_attributes_list = self._get_edge_attributes_from_relations()
        else:
            edge_attributes_list = self._get_edge_attributes_from_reactions()

        # Prioritize reaction edges over relations by populating `edge_attributes` with reaction edges first
        for edge_attributes in edge_attributes_list: 

            # Get edges in `edge_attributes` as a list of sets
            existing_edges = [set([edge_attributes['source'], edge_attributes['target']]) for edge_attributes in edge_attributes_list]

            # Add edge attribute if the edge has not been seen before
            if set([edge_attributes['source'], edge_attributes['target']]) not in existing_edges: 

                edge_attributes_list.append(edge_attributes)

        # Convert to DataFrame and replace group edges 
        edge_attributes_df = pd.DataFrame(edge_attributes_list, columns=self.edge_columns)
        edge_attributes_df = self._replace_group_edges(edge_attributes_df,replace_with_official_gene = replace_with_official_gene)

        return edge_attributes_df


    def _get_directed_edge_attributes_as_dataframe(self, edge_attributes_df): 
        # If A<-->B, this function splits into two relations: A-->B and B-->A
        if len(edge_attributes_df) == 0: return edge_attributes_df

        reverse_edges_df = edge_attributes_df[edge_attributes_df['effect'].isin([-2,0,2])].rename(columns={ 'source': 'target', 'target': 'source'})
        directed_edge_attributes_df = pd.concat([edge_attributes_df, reverse_edges_df], sort=False)

        return directed_edge_attributes_df


    def _populate_edge_attributes(self, source, target, edge_type, interactions): 

        # Attribute `effect` takes values 0 (---), 1 (-->), 2 (<->), or -1 (--|) to indicate cases where
        # orientation is unknown, the edge is activating, the edge is bidirectional (protein complex), or the edge is inhibitory.
        # Perhaps add `binding` as an attribute? Interactions? 
        edge_attributes = { 'source': source, 'target': target, 'type': edge_type, 
                            'effect': 0, 'indirect': 0, 'modification': "" }

        # Attributes must be updated in two steps, since descriptors examined in the second loop
        # are more specific than those in the first, and should be used to overwrite them. 
        for interaction in interactions: 

            if   interaction == 'binding/association': edge_attributes.update({ 'effect': 2 })
            elif interaction == 'protein complex':     edge_attributes.update({ 'effect': 2 }) # not standard type, but including for clarity
            elif interaction == 'bidirected':          edge_attributes.update({ 'effect': 2 }) # not standard type, but including for clarity
            elif interaction == 'dissociation':        edge_attributes.update({ 'effect': 1 })
            elif interaction == 'missing interaction': edge_attributes.update({ 'effect': 0 })
            elif interaction == 'indirect effect':     edge_attributes.update({ 'effect': 1, 'indirect': 1 })
            else: pass

        for interaction in interactions: 

            if   interaction == 'phosphorylation':   edge_attributes.update({ 'effect': 1, 'modification': "+p" })
            elif interaction == 'dephosphorylation': edge_attributes.update({ 'effect': 1, 'modification': "-p" })
            elif interaction == 'glycosylation':     edge_attributes.update({ 'effect': 1, 'modification': "+g" })
            elif interaction == 'ubiquitination':    edge_attributes.update({ 'effect': 1, 'modification': "+u" })
            elif interaction == 'methylation':       edge_attributes.update({ 'effect': 1, 'modification': "+m" })

        for interaction in interactions: 

            if   interaction == 'activation':     edge_attributes.update({ 'effect':  1 })
            elif interaction == 'inhibition':     edge_attributes.update({ 'effect': -1 })
            elif interaction == 'expression':     edge_attributes.update({ 'effect':  1, 'modification': 'e'})
            elif interaction == 'repression':     edge_attributes.update({ 'effect': -1, 'modification': 'e'})
            else: pass

        return edge_attributes


    def _get_edge_attributes_from_reactions(self): 

        reaction_attributes_list = []

        for reaction in self._reactions: 

            compound_id, reaction_name, reaction_type = reaction.get('id'), reaction.get('name'), reaction.get('type')
            substrate_ids = [substrate.get('id') for substrate in reaction.findall('substrate')]
            product_ids   = [product.get('id')   for product   in reaction.findall('product')]

            # Add substrate-compound interactions first 
            for substrate_id in substrate_ids: 

                if reaction_type == 'irreversible': 
                    reaction_attributes_list.append(self._populate_edge_attributes(substrate_id, compound_id, reaction_name, ['activation']))
                else: 
                    reaction_attributes_list.append(self._populate_edge_attributes(compound_id, substrate_id, reaction_name, ['activation']))

            # Add compound-product interactions next
            for product_id in product_ids: 
                
                reaction_attributes_list.append(self._populate_edge_attributes(compound_id, product_id, reaction_name, ['activation']))

        return reaction_attributes_list


    def _get_edge_attributes_from_relations(self): 

        relation_attributes_list = []

        for relation in self._relations: 

            source, target = relation.get('entry1'), relation.get('entry2')
            edge_type = relation.get('type')
            edge_descriptors = [subtype.get('name') for subtype in relation.findall('subtype')]

            # TODO: add support for maplinks?
            if edge_type in [ 'ECrel', 'PPrel', 'GErel', 'PCrel']: 

                if 'compound' not in edge_descriptors: 

                    relation_attributes_list.append(self._populate_edge_attributes(source, target, edge_type, edge_descriptors))

                else: 
                    # Get compound id by first searching through subtypes with `name` attribute equal to 'compound', then retrieving 'value'
                    compound_id = relation.find('.//subtype[@name="compound"]').get('value')

                    relation_attributes_list.append(self._populate_edge_attributes(source, compound_id, edge_type, edge_descriptors))
                    relation_attributes_list.append(self._populate_edge_attributes(compound_id, target, edge_type, edge_descriptors))

        return relation_attributes_list


    def _replace_group_edges(self, edge_attributes_df, replace_with_official_gene=True):
            """Mapping official gene names."""
	   
            id_to_gene_name = {}
            for entry in self._entries:
                if entry.attrib.get('type') == 'gene':
                    graphics = entry.find('graphics')
                    if graphics is not None:
                        gene_name = graphics.attrib.get('name', 'N/A').split(', ')[0].rstrip('.')
                        entry_id = entry.attrib['id']
                        id_to_gene_name[entry_id] = gene_name  # 대문자로 저장
            #print("ID to Gene Name mapping created:", {k: v for k, v in list(id_to_gene_name.items())[:5]})  # 디버깅 출력
    
            for group_element in self._groups:
                group_id = group_element.get('id')
                group_members = [component.get('id') for component in group_element.findall('component')]
                for node_type in ['source', 'target']:
                    edges_with_df = edge_attributes_df[edge_attributes_df[node_type] == group_id]
                    edges_without_df = edge_attributes_df[edge_attributes_df[node_type] != group_id]
                    expanded_edges_df = pd.concat([edges_with_df] * len(group_members)).sort_index()
                    expanded_edges_df[node_type] = group_members * len(edges_with_df)
                    edge_attributes_df = pd.concat([expanded_edges_df, edges_without_df]).reset_index(drop=True)
    
                existing_edges = set(tuple(sorted([row['source'], row['target']])) for _, row in edge_attributes_df.iterrows())
                group_rows = [
                    row for row in [self._populate_edge_attributes(a, b, 'PComplex', ['protein complex']) for a, b in combinations(group_members, 2)]
                    if tuple(sorted([row['source'], row['target']])) not in existing_edges
                ]
                if group_rows:
                    edge_attributes_df = pd.concat([edge_attributes_df, pd.DataFrame(group_rows, columns=self.edge_columns)], ignore_index=True).fillna(0)
    
            # Substitution by official gene names for scRNA-seq analysis
            # Unmapped IDs (e.g., group or non-gene IDs) are preserved.
            if replace_with_official_gene:
                for col in ['source', 'target']:
                    edge_attributes_df[col] = edge_attributes_df[col].map(lambda x: id_to_gene_name.get(x, x))
                    
            return edge_attributes_df


    #### OUTPUTS ####

    def output_KGML_as_full_networkx(self): 

        pass

    def output_KGML_as_directed_networkx(self, genes_only=True): 

        directed_edge_attributes_df = self._get_directed_edge_attributes_as_dataframe(self.edge_attributes_df)

        if genes_only: 

            inferred_edge_attributes_df = self._get_directed_edge_attributes_as_dataframe(self.inferred_edge_attributes_df)
            directed_edge_attributes_df = pd.concat([directed_edge_attributes_df, inferred_edge_attributes_df])

            graph = nx.from_pandas_edgelist(directed_edge_attributes_df, 'source', 'target', edge_attr=True, create_using=nx.DiGraph())
            graph = nx.DiGraph(graph.subgraph(self.node_attributes_df.index[self.node_attributes_df['type'] == 'gene']))

        else: 
            
            graph = nx.from_pandas_edgelist(directed_edge_attributes_df, 'source', 'target', edge_attr=True, create_using=nx.DiGraph())

        nx.set_node_attributes(graph, self.entry_attributes_df.to_dict('index'))
        nx.relabel_nodes(graph, { node_id: graph.node[node_id]['name'] for node_id in graph.nodes() }, copy=False)
        graph.name = self.name

        return graph


    def get_directed_edges_from_KGML(self, genes_only=True): 

        graph = self.output_KGML_as_directed_networkx(genes_only)
        edge_attributes_df = nx.to_pandas_edgelist(graph)
        edge_attributes_df['pathway'] = self.name

        return edge_attributes_df


    def output_KGML_as_graphml(self, path, visualize='full'): 
        """
        Outputs KGML as graphml for visualization in Cytoscape. There are three visualizations which may be set via
        the `visualize` argument. 

        Note: 
            This function is slightly different than writing the output of `self.output_KGML_as_directed_networkx`, which
            represents bidirectional edges as two distinct directed edges. Furthermore, nodes in `self.output_KGML_as_directed_networkx` 
            are labeled using their gene symbols. Lastly, `self.output_KGML_as_directed_networkx` does not support full KEGG visualization. 

        Arguments: 
            path (str): output path
            visualize (str): specifies visualization mode
                'full': displays all KEGG entries, including maps, titles, compounds, etc. 
                'biomolecules': displays only genes and compounds
                'genes': displays only genes

        Returns: 
            path
        """

        # Initialize graph from `edge_attributes_df`, making sure empty dataframes are initialized properly.
        if len(self.edge_attributes_df) > 0:
            graph = nx.from_pandas_edgelist(self.edge_attributes_df.fillna(''), 'source', 'target', edge_attr=True, create_using=nx.DiGraph())
        else: 
            graph = nx.DiGraph()

        # Detailed visualization includes singletons as non-gene or compound nodes, such as orthology, titles, etc.
        if visualize == 'full': 
            graph.add_nodes_from(self.entry_attributes_df.index)
            graph.name = self.name + '_full_KEGG'

        elif visualize == 'biomolecules': 
            graph.name = self.name + '_genes_compounds'

        elif visualize == 'genes': 
            # Create a graph with inferred edges between genes, then add those edges to graph
            inferred_edges_graph = nx.from_pandas_edgelist(self._infer_gene_edges_from_reactions(), 'source', 'target', edge_attr=True, create_using=nx.DiGraph())
            graph = nx.compose(graph, inferred_edges_graph)
            # Remove any compound nodes by selecting only genes
            graph = nx.DiGraph(graph.subgraph(self.node_attributes_df.index[self.node_attributes_df['type'] == 'gene']))
            graph.name = self.name + '_genes_only'

        else: pass

        nx.set_node_attributes(graph, self.entry_attributes_df.to_dict('index'))

        nx.write_graphml(graph, path)

        return path


    ## VISUALIZE

    plt.rcParams['font.family'] = 'DejaVu Sans'

    def view(self, scale=1, show_compounds=False, gene_values=None,replace_with_official_gene=True):
        """Visualize KEGG pathway.

        Arguments:
            scale (float): Scaling factor for plot size.
            show_compounds (bool): Include compounds in visualization.
            gene_values (pandas.Series): Gene-specific values for coloring.

        Returns:
            tuple: (fig, ax) Matplotlib figure and axes.
        """
        # Filter entries
        entry_attributes_df = self.entry_attributes_df.replace('', np.nan).dropna(subset=['name'])
        types = ['gene','compound','map','group'] if show_compounds else ['gene', 'map', 'group']
        entry_attributes_df = entry_attributes_df[entry_attributes_df['type'].isin(types)]

        # Set up plot
        fig, ax = set_grid(
            xlim=entry_attributes_df.x.agg([np.min, np.max]).tolist(),
            ylim=entry_attributes_df.y.agg([np.min, np.max]).tolist(),
            scale=scale
        )

        # Gene coloring
        gene_colors = {}
        if gene_values is not None:
            hex_lookup = [color for color in sns.color_palette("coolwarm", 256).as_hex()] + ['#b3b3b3']
            available_names = entry_attributes_df['name'].unique()
            gene_values = gene_values.reindex(available_names).dropna()
            rgb_indices = ((gene_values / gene_values.abs().max() + 1) / 2 * 255).fillna(-1).astype(int)
            gene_colors = {node: hex_lookup[idx] for node, idx in rgb_indices.items()}

        # Create nodes
        
        if replace_with_official_gene == False:
            nodes_dic = {node_id: Node(attribs)
                     for node_id, attribs in entry_attributes_df.to_dict('index').items()}
        else:
            nodes_dic = {attribs['name']: Node(attribs) for node_id, attribs in entry_attributes_df.to_dict('index').items()}
            id_to_name = {row['id']: row['name'] for _, row in entry_attributes_df.reset_index().iterrows()}
            
        # Render nodes
        for node_id, node in nodes_dic.items():
            
            if replace_with_official_gene:
                idx = entry_attributes_df.index[entry_attributes_df['name'] == node_id][0]
            else:
                idx = node_id
            bgcolor = gene_colors.get(entry_attributes_df.loc[idx, 'name'], '#b3b3b3') if gene_values is not None else entry_attributes_df.loc[idx, 'bgcolor']
            fgcolor = entry_attributes_df.loc[idx, 'fgcolor']
            label = entry_attributes_df.loc[idx, 'name'][6:] if entry_attributes_df.loc[idx,'type'] == 'map' and entry_attributes_df.loc[idx, 'name'].startswith('TITLE:') else entry_attributes_df.loc[idx, 'name']
            fontsize = max(4, min(10,6*scale*(node.width/50)))
            
            if node.shape == 'circle':
                patch = Circle(node.center, node.width / 2, facecolor=bgcolor, edgecolor=fgcolor)
            else:  # rectangle

                patch = Rectangle(node.anchor, node.width, node.height, facecolor=bgcolor, edgecolor=fgcolor)
            ax.add_patch(patch)


            if node.shape == 'circle':
                ax.text(node.center[0], node.center[1], label, ha = 'center', va='center', fontsize = fontsize)
            else:
                ax.text(node.center[0], node.center[1] + node.height/8, label, ha = 'center', va = 'center', fontsize=
                        fontsize)
        
            
        # Render edges
        for _, edge_attribs in self.edge_attributes_df.iterrows():
            source_name = edge_attribs['source']
            target_name = edge_attribs['target']
            if self.replace_with_official_gene:
                source_name = id_to_name.get(source_name, source_name)
                target_name = id_to_name.get(target_name, target_name)
            if source_name in nodes_dic and target_name in nodes_dic:
                source_node = nodes_dic[source_name]
                target_node = nodes_dic[target_name]
                # Use shortest_arrow to calculate optimal positions
                source_pos, target_pos = shortest_arrow(source_node, target_node)
                arrowprops = dict(color = 'k')
                
                if edge_attribs['indirect'] == 1: arrowprops['linestyle'] = '--'
                elif edge_attribs['indirect'] == 0: arrowprops['linestyle'] = '-'
                else: arrowprops['linestyle'] = '-'
                
                if edge_attribs['effect'] == 1:
                	arrowprops['arrowstyle'] = '-|>'
                elif edge_attribs['effect'] == 2:
                	arrowprops['arrowstyle'] = '<|-|>'
                elif edge_attribs['effect']== -1:
                	arrowprops['arrowstyle'] = '|-|, widthA=0, widthB=0.5'
                else:
                	arrowprops['arrowstyle'] = '-'
                
                ax.annotate('', xy=target_pos, xytext=source_pos, arrowprops=arrowprops)
                
                if edge_attribs['modification'] != '':
                	midpoint = (source_pos + target_pos)/2
                	ax.text(x = midpoint[0], y=midpoint[1] -5, s=edge_attribs['modification'], color = 'red', ha = 'center', va = 'center', fontsize = 6*scale)
                
        return fig, ax



def output_DiGraph_as_graphml(graph, path): 
    """
    Removes bidirectional edges from networkx DiGraph for visualization in Cytoscape. 

    Arguments: 
        graph (networkx.DiGraph): any instance of networkx.Graph
        path (str): output path

    Returns: 
        path
    """

    graph_out = graph.copy()

    # For an edge A-->B in the graph, if B-->A is also in the graph, remove A-->B
    for source,target in list(graph_out.edges): 
        if graph_out.has_edge(target, source): graph_out.remove_edge(source, target)

    nx.write_graphml(graph_out, path)

    return path
