#!/usr/bin/env python
"""
A script to take a list of drosophila genes, search flybase to get all pseudonyms,
extract all literature related to the genes, and construct a network showing the
relationship between genes.
"""

import argparse
import itertools
import os
import re
from subprocess import Popen, PIPE
import sys
import tempfile

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans

import config

######################################################################

def define_arguments():
    parser = argparse.ArgumentParser(description=
            """A script to take a list of drosophila genes, search flybase to get all
            pseudonyms, extract all literature related to the genes, and construct a
            network showing the relationship between genes.""")

    ### input options ###
    # logging options:
    parser.add_argument("-q", "--quiet", action='store_true',default=False,
                        help="print fewer messages and output details")
    parser.add_argument("-d", "--directory", type=str,
                        help="specify the directory to save results to")
    parser.add_argument("-o", "--output", type=str, default='text_mining',
                        help="specify the filename to save results to")
    parser.add_argument("-D", "--display_on", action='store_true',default=False,
                        help="display graph results (eg for p value calculation)")


    # data input options:
    parser.add_argument("gene", type=str, nargs='+',
                        help="A flybase gene id (format FBgn0000000")

    # analysis options:
    parser.add_argument('-n', "--network", type=int, default=1,
                        help="""Type of network to draw. 1=circular, 2=random, 3=spectral,
                         4=spring, 5=shell, 6=pygraphvis
                        """)
    parser.add_argument("--max", type=float, default=20,
                        help="""Threshold for solid bars to display in network
                        [default = 20]""")
    parser.add_argument("--min", type=float, default=5,
                        help="""Minimum network weight to display [default=5]""")
    parser.add_argument('-l', "--lsa", type=int,
                        help="""number of components to use for latent semantic analysis.
                        2 is good for visualization, 100 is good for analysis.
                        """)
    parser.add_argument('-c', "--n_clusters", type=int, default=4,
                        help="""Number of clusters to use.
                        """)

    return parser


def get_fly_papers(gene):
    url = "http://flybase.org/cgi-bin/uniq.html?query=[FBgn-FBID_KEY:%s]%%3Efbrf" % gene
    p1 = Popen(["wget", url, '-O', '-'], stdout=PIPE, stderr=PIPE)
    return p1

def get_fly_reference(refid):
    url = "http://flybase.org/reports/%s" % refid
    p1 = Popen(["wget", url, '-O', '-'], stdout=PIPE, stderr=PIPE)
    return p1

def get_refs_from_paper(flyhandle):
    p1 = Popen(["grep", "-o", 'FBrf\S*html'], stdin=flyhandle.stdout, stdout=PIPE)
    flyhandle.stdout.close()
    refs = p1.communicate()[0]
    return set(refs.split('\n')) - set([""])

def parse_ref_page(refhandle):
    p1 = Popen(["grep", '-A', '50', 'field_label">Citation'], stdin=refhandle.stdout, stdout=PIPE)
    #p2 = Popen(["grep", "-A", '20', 'field_label">Abstract'], stdin=refhandle.stdout, stdout=PIPE)
    refhandle.stdout.close()

    # parse reference:
    ref = p1.communicate()[0]  # returns the first match
    citation, doi, abstract = clean_citation(ref)

    return citation, doi, abstract

def clean_citation(html_text):
    #r_search = re.search('(?:<td>)?([^<>]+)<a href="([^"<>]+)">(?:<.*?>)?([^<>"]*)', html_text)
    html_text = html_text.replace('\n', '')
    cit_search = re.search('Citation</th>\s*<td>(.*?)</td>', html_text)
    doi_search = re.search('(http://dx.doi.org[^\"]*)', html_text)
    txt_search = re.search('Abstract</th>\s*<td>(.*?)</td>', html_text)
    com_search = re.search('Communication</th>\s*<td>(.*?)</td>', html_text)
    if cit_search:
        citation = re.sub('<.*?>', '', cit_search.group(1))
        clean_citation = re.sub('\s\s+', '', citation)
    else:
        clean_citation = None

    if doi_search:
        doi = doi_search.group(1)
    else:
        doi = None

    if txt_search:
        abstract = txt_search.group(1)
        clean_abstract = re.sub('\s\s+', " ", abstract)
    else:
        clean_abstract = None

    if com_search:
        coms = com_search.group(1)
        clean_coms = re.sub('\s\s+', " ", coms)
        cleaner_coms = re.sub('<.*?>', "", clean_coms)
    else:
        cleaner_coms = None

    if clean_abstract and len(clean_abstract) > 5:
        txt = clean_abstract
    else:
        txt = cleaner_coms

    return clean_citation, doi, txt

def draw_network(network_weights, report=None, maxweight=20, minweight=5, display=True,
                network_type=1):
    df = pd.DataFrame(network_weights).fillna(0)

    cluster_array = df.values
    dt = [('len', float)]
    cluster_array = cluster_array.view(dt)

    # create networkx object:
    G = nx.from_numpy_matrix(cluster_array)
    relabel = dict(zip(range(len(G.nodes())),df.columns))
    G = nx.relabel_nodes(G, relabel)

    # create dictionary to convert names back to positional labels:
    #backlabels = dict(zip([ clean_filename(f) for f in df.columns], range(len(G.nodes()))))
    #print backlabels.items()[:5]

    # add weights to each edge for later processing:
    e = [ (n1, n2, G[n1][n2]['len']) for n1 in G for n2 in G[n1]   ]
    G.add_weighted_edges_from(e)

    # define the type of graph to be drawn:
    network_types = {1: nx.circular_layout,
                2: nx.random_layout,
                3: nx.spectral_layout,
                4: nx.spring_layout,
                5: nx.shell_layout,
                6: nx.pygraphviz_layout}
    net_type = network_types[network_type]

    # split into all sub-networks and draw each one:
    pos=net_type(G)
    C = nx.connected_component_subgraphs(G)

    if report:
        pp = PdfPages( report[:-3] + 'pdf' )
    for g in C:
        # report size of sub-network
        verbalise("Y", "%d clusters in sub-network" % (len(g)) )

        # define which edges are drawn bold:
        rlarge =  [(u,v,d) for (u,v,d) in g.edges(data=True) if d['weight'] >= maxweight]
        rmedium =[(u,v,d) for (u,v,d) in g.edges(data=True) if maxweight > d['weight'] >= minweight]
        rsmall =  [(u,v,d) for (u,v,d) in g.edges(data=True) if d['weight'] < minweight]

        elarge =  [ (u,v) for (u,v,d) in rlarge  ]
        emedium = [ (u,v) for (u,v,d) in rmedium ]
        esmall =  [ (u,v) for (u,v,d) in rsmall  ]

        rlarge.sort(key=lambda x: x[2]['weight'])
        rmedium.sort(key=lambda x: x[2]['weight'])
        rsmall.sort(key=lambda x: x[2]['weight'])

        # report number of clusters with each weight
        verbalise("M",
            "%d cluster pairs with %d or more shared members" % (len(elarge),maxweight))
        verbalise("G",
            "\n".join([ "%-6r %-6r %s" % (t[0], t[1], t[2]['weight']) for t in rlarge]))

        verbalise("M",
            "%d cluster pairs with less than %d and %d or more shared members" %
            (len(emedium),maxweight, minweight))
        verbalise("G",
            "\n".join(["%-6r %-6r %s" % (t[0], t[1], t[2]['weight']) for t in rmedium][-3:]))

        verbalise("M",
            "%d cluster pairs with less than %d shared members" % (len(esmall),minweight))
        verbalise("G",
            "\n".join(["%-6r %-6r %s" % (t[0], t[1], t[2]['weight']) for t in rsmall][-3:]))
        verbalise("G","")

        if report:
            handle = open(report, 'a')

            handle.write("%d clusters in sub-network\n" % (len(g)))
            handle.write(
                "%d cluster pairs with %d or more shared members\n" % (len(elarge),maxweight))
            handle.write(
                "%d cluster pairs with less than %d and %d or more shared members\n" %
                (len(emedium),maxweight, minweight)
                        )
            handle.write(
                "%d cluster pairs with less than %d shared members\n" % (len(esmall),minweight))
            handle.write(
                "\n".join(["%-6r %-6r %s" % (t[0], t[1], t[2]['weight']) for t in rlarge])
                 + '\n')
            handle.write(
                "\n".join(["%-6r %-6r %s" % (t[0], t[1], t[2]['weight']) for t in rmedium])
                 + '\n')
            handle.write(
                "\n".join(["%-6r %-6r %s" % (t[0], t[1], t[2]['weight']) for t in rsmall])
                 + '\n\n')
            handle.close()


        # draw edges (partitioned by each edge type):
        # large:
        nx.draw_networkx_edges(g,pos,edgelist=elarge,
                width=2, edge_color='purple')
        # medium:
        nx.draw_networkx_edges(g,pos,edgelist=emedium,
                width=1, alpha=0.3, edge_color='blue')
        # small:
        #nx.draw_networkx_edges(g,pos,edgelist=esmall,
        #        width=1, alpha=0.0, edge_color='blue', style='dashed')

        # draw sub-network:
        nx.draw(g,
             pos,
             node_size=40,
             node_color='g',
             vmin=0.0,
             vmax=2.0,
             width=0,
             with_labels=True
             )
        if report:
            plt.savefig(pp, format='pdf')
        if display:
            plt.show()
        else:
            plt.close()

    if report:
        pp.close()
    return rlarge, rmedium, rsmall

def cluster_abstracts(genes, common_papers, paper_details, n_clusters=4,
                        min_text_num=50, lsa_size=100):
    """
    Clustering of abstracts that are common between any pair of genes
    """
    labels = { }
    for gene1,gene2 in  itertools.combinations(genes, r=2):
        if len(common_papers[gene1][gene2]) >= min_text_num:
            texts = [ paper_details[t][2] for t in common_papers[gene1][gene2]
                        if paper_details[t][2] ]

            verbalise("B",
            "\rExtracting features from %d abstracts using a sparse vectorizer" % len(texts))
            vectorizer = TfidfVectorizer(max_df=0.95, max_features=10000,
                                                 min_df=2, stop_words='english',
                                                 use_idf=True)
            tfidf = vectorizer.fit_transform(texts)
            if lsa_size:
                verbalise("B", "Performing dimensionality reduction using LSA")
                svd = TruncatedSVD(lsa_size)
                normalizer = Normalizer(copy=False)
                lsa = make_pipeline(svd, normalizer)

                X = lsa.fit_transform(tfidf)

                explained_variance = svd.explained_variance_ratio_.sum()

            km = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=1,
                     init_size=1000, batch_size=1000, verbose=False)

            verbalise("B", "Clustering sparse data with %s" % km)
            km.fit(X)


            if lsa_size:
                original_space_centroids = svd.inverse_transform(km.cluster_centers_)
                order_centroids = original_space_centroids.argsort()[:, ::-1]
            else:
                order_centroids = km.cluster_centers_.argsort()[:, ::-1]

            terms = vectorizer.get_feature_names()

            labels[gene1,gene2] = (km.predict(X),
                                    [ paper_details[t] for t in common_papers[gene1][gene2]
                                        if paper_details[t][2] ],
                                    order_centroids,
                                    terms,
                                    )

    return labels

############################################################

if __name__ == '__main__':
    parser = define_arguments()
    args = parser.parse_args()

    verbalise = config.check_verbose(not(args.quiet))
    logfile = config.create_log(args, outdir=args.directory, outname=args.output)

    temp_dir = tempfile.mkdtemp()

    # extract the flybase reference ids for all genes provided:
    verbalise("B", "Collecting references from flybase...")
    fly_refs = {}
    for gene in args.gene:
        flyhandle = get_fly_papers(gene)
        fly_refs[gene] = get_refs_from_paper(flyhandle)

    # find number of overlapping references between each gene (for network graph):
    network_weights = {g:{g2:0.0 for g2 in args.gene} for g in args.gene}
    common_papers = {g:{g2:0.0 for g2 in args.gene} for g in args.gene}
    paper_set = set([])
    for node1,node2 in itertools.combinations(args.gene, r=2):
        network_weights[node1][node2] = len(fly_refs[node1] & fly_refs[node2])
        common_papers[node1][node2] = fly_refs[node1] & fly_refs[node2]
        if network_weights[node1][node2] >= args.max:
            paper_set = paper_set |  common_papers[node1][node2]


    verbalise("B", "Drawing network graphs")
    # draw network:
    draw_network(network_weights,
                report="%sout" % logfile[:-3],
                maxweight=args.max,
                minweight=args.min,
                display=args.display_on,
                network_type=args.network)

    # extract references for strongest links:
    verbalise("B", "Downloading and extracting %d paper references" % len(paper_set))
    paper_details = {}

    for i, p in enumerate(paper_set):
        refhandle = get_fly_reference(p)
        citation, doi, abstract = parse_ref_page(refhandle)
        paper_details[p] = (citation, doi, abstract)

        if i % 15 == 0:
            sys.stdout.write("\r%d downloaded..." % i )
            sys.stdout.flush()
        #verbalise("Y", title)


    handle = open("%scommon_papers.txt" % logfile[:-3], 'w')
    for gene1,gene2 in  itertools.combinations(args.gene, r=2):
        if len(common_papers[gene1][gene2]) >= args.max:
            handle.write("\n### %s %s ###\n" % (gene1, gene2))
            for cp in common_papers[gene1][gene2]:
                citation, doi, abstract = paper_details[cp]
                handle.write(
                    "http://flybase.org/reports/%s\n%s %s\n%s\n\n" % (cp,
                                                                    citation,
                                                                    doi,
                                                                    abstract)
                            )

    handle.close()

    # clustering of abstracts!
    labels = cluster_abstracts(args.gene, common_papers, paper_details, n_clusters=args.n_clusters,
                        min_text_num=args.max, lsa_size=args.lsa)

    handle = open("%stext_clustering.out" % logfile[:-3], 'w')
    for g1,g2 in labels:
        verbalise("R",  "Clustering of common papers between %s and %s" % (g1, g2))
        handle.write("### Clustering of common papers between %s and %s\n" % (g1, g2))
        clusters = sorted(zip(labels[g1,g2][0],labels[g1,g2][1]), key=lambda x:x[0])
        for i in range(max(labels[g1,g2][0]) + 1):
            verbalise("M",
                ">Cluster %d (%d papers)" % (i,
                                            sum(1 for p in labels[g1,g2][0] if p == i) ))
            verbalise("B", "most important terms in cluster:")
            handle.write("Cluster %d\nmost important terms in cluster:\n" % i)
            for ind in labels[g1,g2][2][i, :10]:
                verbalise("C", '   %s' % labels[g1,g2][3][ind])
                handle.write('   %s' % labels[g1,g2][3][ind])
            print
            cdocs = [ d for d in clusters if d[0] == i ]
            for doc in cdocs:
                #verbalise("G", doc[1][0])
                #verbalise("Y", doc[1][2])
                #print ""
                handle.write( "%s\n%s\n%s\n\n" % (doc[1][0], doc[1][1], doc[1][2]))
    handle.close()

    # clean up temp files and directory
    for file in [ "notheere", "alsonothere" ]:
        if os.path.exists(file):
            os.remove(file)

    os.rmdir(temp_dir)  # dir must be empty!
