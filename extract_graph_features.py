# -*- coding: utf-8 -*-
"""

@author: Samip
"""
class GraphFeatures:

    def extractGraphFeatures(password):

        """ADDING GRAPH FEATURES"""
        from py2neo import Graph

        graph = Graph("http://35.236.81.95:7474/db/data/", user="neo4j", password=password)

        try:
            graph.run("Match () Return 1 Limit 1")
        except Exception as e:            
            return "failure: ",e
            
        # Helper functions to add network features to input dataframe 
        def add_degree(x):
            return valueDict[x]['degree']
        def add_community(x):
            return str(valueDict[x]['community']) # cast to string for one-hot encoding
        def add_pagerank(x):
            return valueDict[x]['pagerank']
        
        # Read in a new dataframe and add network features 
        import pandas as pd
        df = pd.read_csv("data/bs140513_032310.csv")
        
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
        
        df.to_csv('data/bs140513_032310_graphed.csv',index=False)

        return "success"

