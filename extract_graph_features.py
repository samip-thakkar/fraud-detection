# -*- coding: utf-8 -*-
"""

@author: Samip
"""
class GraphFeatures:

    def extractGraphFeatures(self):
        import pandas as pd
        df = pd.read_csv("bs140513_032310.csv")
        
        """ADDING GRAPH FEATURES"""
        from py2neo import Graph
        graph = Graph(password = ' ', bolt_port = 7687, http_port = 7474)
        
            
        # Helper functions to add network features to input dataframe 
        def add_degree(x):
            return valueDict[x]['degree']
        def add_community(x):
            return str(valueDict[x]['community']) # cast to string for one-hot encoding
        def add_pagerank(x):
            return valueDict[x]['pagerank']
        
        # Read in a new dataframe and add netork features 
        import pandas as pd
        df = pd.read_csv("bs140513_032310.csv")
        
        query = """
        MATCH (p:Placeholder)
        RETURN p.id AS id, p.degree AS degree, p.pagerank as pagerank, p.community AS community 
        """
        
        data = graph.run(query)
        valueDict = {}
        for d in data:
            valueDict[d['id']] = {'degree': d['degree'], 'pagerank': d['pagerank'], 'community': d['community']}
        
        #Append the graph featuers to the dataframe
        df['merchDegree'] = df.merchant.apply(add_degree)
        df['custDegree'] = df.customer.apply(add_degree)
        df['custPageRank'] = df.customer.apply(add_pagerank)
        df['merchPageRank'] = df.merchant.apply(add_pagerank)
        df['merchCommunity'] = df.merchant.apply(add_community)
        df['custCommunity'] = df.customer.apply(add_community)
        #Save thenew data
        df.to_csv('data_with_graph_features.csv')
