import numpy as np
import pandas as pd
import AMP_lib as amp

__AUTHOR__  = "Boris Sotomayor Gómez"
__EMAIL__   = "bsotomayor92@gmail.com"
__CREATED__ = "22/07/18"
__VERSION__ = "01/08/18"

"""
-----------------------------------------------------------------------------------------
"""
# global variables
FIG_FOLDER = "" # folder to store each plot

def doTheAnalysis(dsname, n_max_tokens, n_topics, n_max_tokens_toprint,LDA_max_iter, LDA_learning_offset, LDA_random_state, ngram_range, lemmatize, topic_labels=None, format=None):
	# LECTURA DE DATOS
	print ("[INFO] - Leyendo de dataset")
	raw_data = pd.read_csv('../Beca1/dataset/sophia_%s_v2.csv' % dsname, sep='\n', header=None)
	"""n_max_tokens = 5000
	n_topics = 5
	n_max_tokens_toprint = 15
	LDA_max_iter = 30
	LDA_learning_offset = 50.
	LDA_random_state = 10
	ngram_range=(1, 2)
	dsname = "mapuche" """

	# DESCARGA STOP-WORDS
	# obtener stopwords y aplicar filtro sobre el texto
	print ("[INFO] - Obtención de stop-words y filtros")
	docs, news_df2, stopwords = amp.stopwordsAndFilters(raw_data, lemmatize=lemmatize)
	# separación de datos de test y train
	# train_proportion = 0.3
	# docs_train = docs[int(len(docs)*train_proportion):]
	# docs_test  = docs[:int(len(docs)*train_proportion)]

	# Vectorización de tokens
	print ("[INFO] - Vectorización de Tokens")
	# tf_train, tf_feature_names_train = vectorizer(n_max_tokens=n_max_tokens, stopwords=stopwords, ngram_range=ngram_range, docs=docs_train)
	# tf, tf_feature_names = vectorizer(n_max_tokens=n_max_tokens, stopwords=stopwords, ngram_range=ngram_range, docs=docs_test)
	tf, tf_feature_names = amp.vectorizer(n_max_tokens=n_max_tokens, stopwords=stopwords, ngram_range=ngram_range, docs=docs)
	
	# Análisis de perplexity
	print ("[INFO] - Cálculo de perplexity")
	amp.plotPerplexity(dsname=dsname, tf=tf, max_topics=15, max_iter=5, learning_offset=50., random_state=50, fig_folder=FIG_FOLDER, format=format)

	# Ejecución de LDA
	print ("[INFO] - Cálculo de LDA")
	W1, W2, components = amp.computeLDA(tf, n_topics, LDA_max_iter, LDA_learning_offset, LDA_random_state)

	# asignar tópico a noticias
	A_lda_keys = []
	for i in range(W2.shape[0]):
		A_lda_keys.append( W2[i].argmax() )

	news_df_topics = news_df2.assign(topic=np.array(A_lda_keys))
	g_topics = news_df_topics.groupby('topic')

	# Filtrar top-20 medios:
	top20_df = (news_df_topics["media"].value_counts()).nlargest(20)#.to_frame()
	top20_list = top20_df.index.values.tolist()
	news_df_topics_top20 = news_df_topics.loc[news_df_topics["media"].isin(top20_list)]
	print(len(news_df_topics_top20["text"]))

	#print ("[INFO] - plotMediaTimeline()")
	#plotMediaTimeline(dsname=dsname, data=W2.T, format=format)

	# Topic content
	# mostrar top-15 palabras de cada tópico
	news_df_topics, topics_list = amp.getNewsWithTopic(tf_feature_names=tf_feature_names, components=components, W2=W2, n_topics=n_topics, news_df_original=news_df2)
	# para mostrar las palabras mas usadas: topics_list[topic_to_analize][0:n_top_topics]
	# [?] print("[????] - Agregar como tabla en material suplementario:")
	# [?] print(news_df_topics["media"].value_counts())

	# Filtrar top-20 medios:
	top20_df = (news_df_topics["media"].value_counts()).nlargest(20)#.to_frame()
	top20_list = top20_df.index.values.tolist()
	news_df_topics_top20 = news_df_topics.loc[news_df_topics["media"].isin(top20_list)]
	#print(len(news_df_topics_top20))

	# plot media "timeline"
	print ("[INFO] - plotMediaTimeline()")
	amp.plotMediaTimeline(dsname=dsname, news_df_topics=news_df_topics, n_topics=n_topics, groupby="week", fig_folder=FIG_FOLDER, format=format, topic_labels=topic_labels)#data=W2.T

	# plot de análisis por medio
	amp.plotTopicByMedia(dsname=dsname, news_df_topics=news_df_topics_top20, n_topics=n_topics, fig_folder=FIG_FOLDER, format=format, topic_labels=topic_labels)
	#plotMediaTimeline(dsname=dsname, data=W2.T, format=format, groupby="week")
	
	#news_df_topics[ news_df_topics['topic' == 0] ]
	print ("[INFO] - printContentByTopic(dsname = %s)" % dsname)
	amp.printContentByTopic(news_df_topics=news_df_topics_top20, topics_list=topics_list, n_topics=n_topics)
	
	#printContentByTopic(tf_feature_names=tf_feature_names, components=components, W2=W2, n_topics=n_topics, news_df_original=news_df2)
	#printContentByTopic(tf_feature_names, components, W2, n_topics, news_df_original)

def main():
	n_max_tokens_toprint = 15
	lemmatize = False

	"""doTheAnalysis(
		dsname = "sexismo",
		n_max_tokens = 5000,
		n_topics = 3,
		topic_labels = [ "sexism & gender", "politics", "declaration/opinion" ],
		n_max_tokens_toprint = n_max_tokens_toprint,
		LDA_max_iter = 30,
		LDA_learning_offset = 50.,
		LDA_random_state = 10,
		ngram_range = (1,2),#(1,2)
		lemmatize=lemmatize,
		format="png"
		)"""
	"""doTheAnalysis(
		dsname = "mapuche",
		n_max_tokens = 5000,
		n_topics = 2,
		n_max_tokens_toprint = n_max_tokens_toprint,
		LDA_max_iter = 30,
		LDA_learning_offset = 50.,
		LDA_random_state = 10,
		ngram_range = (1,1), # En el trabajo revisado aparecía como (1,2).. y el perplexity crecía exponencialmente...
		lemmatize=lemmatize,
		format="png")"""
	"""doTheAnalysis(
		dsname = "cambioclimatico",
		n_max_tokens = 5000,
		n_topics = 5, # DUDA: 3 o 7?
		topic_labels = ['science','events','services','effects','others'],
		n_max_tokens_toprint = n_max_tokens_toprint,
		LDA_max_iter = 30,
		LDA_learning_offset = 50.,
		LDA_random_state = 10,
		ngram_range = (1,1),
		lemmatize=lemmatize,
		format="png"
		)"""
	doTheAnalysis(
		dsname = "feminismo",
		n_max_tokens = 5000,
		n_topics = 2,
		n_max_tokens_toprint = n_max_tokens_toprint,
		LDA_max_iter = 30,
		LDA_learning_offset = 50.,
		LDA_random_state = 10,
		ngram_range = (1,1),
		lemmatize=lemmatize,
		format="png"
		)

if __name__ == '__main__':
	main()