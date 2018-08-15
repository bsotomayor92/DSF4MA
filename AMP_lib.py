import string
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns;
import spacy
import unicodedata
import re
import spacy
import es_core_news_sm

from unicodedata import normalize
from sklearn.decomposition import LatentDirichletAllocation
from stop_words import get_stop_words
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

__AUTHOR__  = "Boris Sotomayor Gómez"
__EMAIL__   = "bsotomayor92@gmail.com"
__CREATED__ = "22/07/18"
__VERSION__ = "01/08/18"

sns.set()
nlp = es_core_news_sm.load()

def getSelectedStopwords(fn):
	l_stopwords = ""
	f = open(fn, "rU")
	for line in f:
		l_stopwords += "%s " % line
	f.close()
	return l_stopwords[:-1]

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def tokenize_lemmatize_filtering(text):
    filtered_lemmas = []
    doc = nlp(text)
    for token in doc :
        if re.search('[a-zA-Z]', token.lemma_.lower()):
            if (token.pos_ == 'NOUN'):# or token.pos_ == 'ADJ' or token.pos_ == 'VERB' ):
                filtered_lemmas.append(token.lemma_.lower())
    return " ".join(filtered_lemmas)

def stopwordsAndFilters(raw_data, lemmatize=False):
	nltk.download('stopwords')
	nltk.download('punkt')
	stemmer = SnowballStemmer("spanish")
	table = str.maketrans('', '', string.punctuation)
	my_stopwords = getSelectedStopwords("selected_stopwords.txt")
	my_stopwords = unicodedata.normalize("NFKD", my_stopwords)

	twitter_stop_words = my_stopwords.lower().split()
	strip_twitter_stop_words = [w.translate(table) for w in twitter_stop_words]

	# FILTRO DE STOP_WORDS
	news = raw_data[0][:]
	news_df = pd.DataFrame(columns=[0, 1, 2])
	news_df_o = pd.DataFrame(columns=[0, 1, 2])
	table = str.maketrans('', '', string.punctuation)

	# Procesamiento de palabras: Eliminación de link, acentos y puntuación (", ; .")
	temp_words_list = []
	for new in news:
		info = new.split('|')
		text = ' '.join(info[2:4])

		tmp_str = unicodedata.normalize("NFKD", text)
		tmp_str_nolinks   = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tmp_str)
		tmp_str_noaccents = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", normalize("NFD", tmp_str_nolinks), 0, re.I)
		tmp_str_noaccents = normalize('NFC', tmp_str_noaccents)

		words = tmp_str_noaccents.lower().split()
		tmp_punctuation = [w.translate(table) for w in words]

		words_alpha = [word for word in tmp_punctuation if word.isalpha()]
		words_no_twitter_stopwords = [w for w in words_alpha if not w in twitter_stop_words]
		temp_words_list.extend(words_no_twitter_stopwords)

		new_text = ' '.join(words_no_twitter_stopwords)
		if (lemmatize): new_text = tokenize_lemmatize_filtering(new_text) # lemmatize
		news_df  = news_df.append(pd.DataFrame(np.array([info[0], info[1], new_text]).reshape([1, 3])))
		news_df_o = news_df_o.append(pd.DataFrame(np.array([info[0], info[1], text]).reshape([1, 3])))

	news_df.reset_index(drop=True, inplace=True)
	news_df_o.reset_index(drop=True, inplace=True)

	news_df.columns = ['timestamp', "media",'text']
	news_df = news_df.sort_values(by='timestamp').reset_index(drop=True)
	
	docs = news_df['text']

	news_df_o.columns = ['timestamp', "media",'text'] # COMENTAR????
	news_df_o['timestamp'] = pd.to_datetime(news_df_o["timestamp"])
	news_df_o = news_df_o.sort_values(by='timestamp').reset_index(drop=True) # COMENTAR????

	# actualización de stop-words
	temp_words = pd.DataFrame(temp_words_list)
	temp_stop_words_min = temp_words.loc[temp_words[0].apply(len) <= 3, 0]
	temp_stop_words_max = temp_words.loc[temp_words[0].apply(len) > 25, 0]

	stopwords = get_stop_words('spanish')
	stopwords.extend(np.array(temp_stop_words_min))
	stopwords.extend(np.array(temp_stop_words_max))

	stopwords.extend(['Tele13_Radio', 'elsoldeiquique', 'vlnradio', 'JGMRadio', 'msncl', 'ElPuelcheChile', 'agenciaunochile', 'RedMiVoz', 'ElGraficoChile', 'InetTvDigital', 'tentos', 'elrancahuaso', 'fisuramagazine', 'paislobo', 'Cooperativa', 'radiomaray', 'tvmaulinos', 'rtierrabella', 'T13', 'TVN', 'pinguinodiario', 'diariolaguino', 'Elperiodicocl', 'FMConquistador', 'LaTerceraTV', 'GAMBA_CL', 'thecliniccl', 'ellanquihue', 'ChillanOnline', 'InforiosCL', 'austral_osorno', 'osornonoticias', 'FortinOficial', 'PublimetroChile', 'AtacamaNoticias', 'ElPeriodista', 'estrellaconce', 'chilevision', 'DigitalFmChile', 'MauleNoticias', 'ColegioProfes', 'El_Ciudadano', 'ConceTv', 'elobservatodo', 'DiarioConce', 'soychilecl', 'soyosorno', 'PrensaChiloe', 'mauriciohofmann', 'cambio21cl', 'elnaveghable', 'pdteibanez', 'PatoFdez', 'CronicaChillan', 'biobio', 'elmostrador', 'soypuertomontt', 'RadioGalactika', 'ElPeriscopioCL', 'lacuarta', 'sancarlosonline', 'diario_eha', 'estrellachiloe', 'bionoticiascl', 'redpanguipulli', 'AoLaonline', 'panguipullinoti', 'rioenlinea', 'CNNChile', 'ladiscusioncl', 'elepicentro', 'CCauqueninoCom', 'carta_abierta', 'LaRedTV', 'elconcecuente', 'Elregionalcl', 'FugadeTinta', 'radiozero977', 'ARAUCOTV', 'MQLTV', 'tvosanvicente', 'elranco', 'Pagina7Chile', 'el_dinamo', 'laestrellaiqq', 'diario_elandino', 'ahoranoticiasAN', '24HorasTVN', 'ForoChile', 'Clave9cl', 'eldesconcierto', 'soychillan', 'rsumen', 'adnradiochile', 'DiarioOvalleHoy', 'NatalesOnLine', 'nacioncl', 'Azkintuwe', 'soytemuco', 'Mapuexpress', 'ELCLARINDECHILE', 'radiosantamaria', 'HDtuiter', 'latercera', 'soyconcepcion', 'LaPrensAustral', 'espaciosfm', 'RadioDuna', 'uchileradio', 'vlanoticia', 'chillanactivo', 'AgriculturaFM', 'girovisualtv', 'panguipuIIi', 'TerraChile', 'ElLongino', 'laestrellavalpo', 'araucanianews', 'eltipografo', 'Medio_a_Medio', 'DiarioLaHora', 'cronicacurico', 'elsiglochile', 'CHVNoticiascl', 'Temucodiario', 'RADIOPALOMAFM', 'EYN_ELMERCURIO', 'Publimetro_TV', 'prensaopal', 'eldefinido', 'La_Segunda', 'la7talca', 'LaSerenaOnline', 'lidersanantonio', 'maipuciudadano', 'AustralTemuco', 'soysanantonio', 'rvfradiopopular', 'laopinon', 'Educacion2020', 'ddivisadero', 'mercuriovalpo', 'difamadores', 'Radiokonciencia', 'ConectaTV2', 'pulso_tw', 'lanalhue', 'TVU_television', 'soyvalparaiso', 'eldia_cl', 'corrupcionchile', 'Cronica_Digital', 'RNuevoMundo', 'elquintopoder', 'Infestudiantes', 'soyiquique', 'DiarioPaillaco', 'radiopudeto', 'laestrelladeqta'])

	return docs, news_df_o, stopwords # news_df2, al parecer, es el dataset con texto original

def vectorizer(n_max_tokens, stopwords, ngram_range, docs):
	tf_vectorizer = CountVectorizer(tokenizer=tokenize_only, stop_words=stopwords, ngram_range=ngram_range)
	tf = tf_vectorizer.fit_transform(docs)
	tf_feature_names = tf_vectorizer.get_feature_names()
	return tf, tf_feature_names

def computeLDA(tf, n_topics, LDA_max_iter, learning_offset=50., random_state=10):
	model = LatentDirichletAllocation(n_components=n_topics, max_iter=LDA_max_iter, learning_method='online', learning_offset=learning_offset, random_state=random_state)
	W1 = model.fit(tf)
	W2 = model.fit_transform(tf)
	components = W1.components_
	return W1, W2, components

def getNewsWithTopic(tf_feature_names, components, W2, n_topics, news_df_original):
	topics_list = []
	n_max_tokens_toprint = 15
	for i in range(n_topics):
	    d3 = pd.Series(components[i].flatten(), index=tf_feature_names).sort_values(ascending=False)
	    topics_list.append(d3)

	A_lda_keys = []
	for i in range(W2.shape[0]):
		A_lda_keys.append( W2[i].argmax() )

	news_df_topics = news_df_original.assign(topic=np.array(A_lda_keys))
	return news_df_topics, topics_list

def printContentByTopic(news_df_topics, topics_list, n_topics):
	print("[INFO](printContentByTopic()) [INI]")
	groupby_topic = news_df_topics.groupby('topic')

	#print("topics_list\n",topics_list)
	#print("n_topics\n",n_topics)

	n_top_topics = 15 # cantidad de palabras a mostrar de cada tópico
	for n in range(0, n_topics):
		print(topics_list[n][0:n_top_topics])
		U = groupby_topic.get_group(n)
		print(U.iloc[2:3,:])
	print("[INFO](printContentByTopic()) [END]")

def plotPerplexity(dsname, tf, max_topics, max_iter, learning_offset, random_state, fig_folder="", format=None):
	#train_proportion = 0.2
	#tf_test  = tf[int(len(tf)*train_proportion):]
	#tf_train = tf[:int(len(tf)*train_proportion)]

	if(format!=None): plt.clf()
	#l_perplexity_train = []
	#l_perplexity_test  = []
	l_perplexity = []
	for i in range(1,max_topics+1):
		print("\t[DBUG] - (plotPerplexity()): (iter %s of %s)" % (i, max_topics))
		n_topics = i
		lda = LatentDirichletAllocation(
			n_components=n_topics, max_iter=max_iter,
			learning_method='online',
			learning_offset=learning_offset,
			random_state=random_state )
		#lda.fit(tf_train)
		lda.fit(tf)

		gamma = lda.transform(tf)
		perplexity = lda.perplexity(tf, gamma)
		
		#gamma_train = lda.transform(tf_train)
		#perplexity_train = lda.perplexity(tf_train, gamma_train)

		#gamma_test = lda.transform(tf_test)
		#perplexity_test = lda.perplexity(tf_test, gamma_test)
		
		l_perplexity.append(perplexity)
		#l_perplexity_train.append(perplexity_train)
		#l_perplexity_test.append(perplexity_test)

	plt.plot(range(1,max_topics+1),l_perplexity,'-o', alpha=0.75)
	#plt.plot(range(1,max_topics+1),l_perplexity_train,'--o', alpha=0.75, label="train")
	#plt.plot(range(1,max_topics+1),l_perplexity_test  ,'-o', alpha=0.75, label="test")
	plt.xlim(1, max_topics)
	plt.xlabel('# topics')
	plt.ylabel('$ Perplexity $')
	plt.xticks(range(1,max_topics+1))

	if format!= None:
		print("[INFO]: Plot Topic by Media '%sperplexity_%s.%s'" % (fig_folder, dsname, format))
		plt.tight_layout()
		plt.savefig("%sperplexity_%s.%s" % (fig_folder, dsname, format))
	else:
		plt.show()
def addMediaInfo(df_news, fn_medias):
    raw_media_data = pd.read_csv(fn_medias, sep=',')
    raw_media_data=raw_media_data.rename(columns = {'CódigoMedio':'media'})
    
    df_news['media'] = df_news['media'].astype(str)
    raw_media_data['media'] = raw_media_data['media'].astype(str)
    df_news['media'] = df_news['media'].apply(lambda m: m.replace(" ",""))
    raw_media_data['media'] = raw_media_data['media'].apply(lambda m: m.replace(" ",""))

    return pd.DataFrame.merge(df_news, raw_media_data, on='media')

def plotMediaBy(df, xkey, ykey, xlabel=None, ylabel=None, topic_labels=None):
	"""
	Plotea dos variables categóricas selecciondas del dataframe df
	"""
	xkey_names = df[xkey].unique()
	ykey_names = df[ykey].unique()
	plt.clf()
	d_tuples = {} #key:(xkey,ykey) val: freq
	for ii, row in df.iterrows():
		if not row[xkey] in d_tuples:
			d_tuples[ row[xkey] ] = {}
		if not row[ykey] in d_tuples[ row[xkey]]:
			d_tuples[ row[xkey] ][ row[ykey] ] = 0
		d_tuples[ row[xkey] ][ row[ykey] ] += 1
    
	pd_plot = (pd.DataFrame.from_dict(d_tuples)).fillna(0)
	pd_plot = pd_plot.div(pd_plot.sum(axis=1), axis=0)
	ax = sns.heatmap(pd_plot, annot=False, cmap="Greens", cbar=False, vmin=0, vmax=1)
	cbar = ax.figure.colorbar(ax.collections[0])
	cbar.set_ticks([.0, 0.25, 0.50, 0.75, 1.0])
	cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
	if (topic_labels!=None):
		if (len(topic_labels)>5):
			rotation=90#'vertical'
		else:
			rotation=0
		xlabel = "$ Topic $"
		ax.set_xticklabels([ "$%s$" % val.replace(" ","\ ") for val in topic_labels ], rotation=rotation)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
    
def plotTopicByMedia(dsname, news_df_topics, n_topics, topic_labels=None, fig_folder="", format=None):
	if(format!=None): plt.clf()
	d_medios={}
	for indice_fila, fila in news_df_topics.iterrows():
		if not fila["media"] in d_medios:
			d_medios[fila["media"]] = [ 0 for val in range(n_topics) ] #dict({ (val,0) for val in range(n_topics) })
		d_medios[fila["media"]][fila['topic']] += 1
	    
	# normalizamos por cantidad de noticias en cada medio
	for medio, news_by_topic in d_medios.items():
		for ii in range(len(news_by_topic)):
			d_medios[medio][ii] = news_by_topic[ii]/float(sum(news_by_topic)) 

	dfx=pd.DataFrame.from_dict(d_medios,orient='index')
	ax = sns.heatmap(dfx, annot=False, cmap="Greens", cbar=False, vmin=0, vmax=1)
	cbar = ax.figure.colorbar(ax.collections[0])
	cbar.set_ticks([.0, 0.25, 0.50, 0.75, 1.0])
	cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

	# set the topic label for the plot
	xlabel = "$Topic\ Index$"
	ylabel = "$Media$"
	if (topic_labels!=None):
		if (len(topic_labels)>5):
			rotation=90#'vertical'
		else:
			rotation=0
		xlabel = "$ Topic $"
		ax.set_xticklabels([ "$%s$" % val.replace(" ","\ ") for val in topic_labels ], rotation=rotation)
		
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)


	if format!= None:
		print("[INFO]: Plot Topic by Media '%smedia_%s.%s'" % (fig_folder, dsname, format))
		plt.tight_layout()
		plt.savefig("%smedia_%s.%s" % (fig_folder, dsname, format))
	else:
		plt.show()

def plotMediaTimeline(dsname, news_df_topics, n_topics, topic_labels=None, groupby=None,fig_folder="", format=None):
	# PLOT: linea de tiempo con presencia de tópicos
	# data to plot
	if topic_labels!=None: topic_labels = topic_labels[::-1]
	temp = news_df_topics.copy()
	xlabel = "$article\ index\ (ordered\ chronologically)$"
	step = 0 # cantidad de dias/semanas/meses a agregar en index
	if  (groupby == 'day'):
		temp['timestamp'] = temp['timestamp'].dt.day
		xlabel = "$week\ index$"
	elif(groupby == 'week'):
		#print ("type=",type(temp['timestamp'].dt.year), "value:", temp['timestamp'].dt.week)
		#if temp['timestamp'].dt.year == 2018:
		temp['timestamp'] = temp['timestamp'].dt.week# + 52
		step = 52
		xlabel = "$week\ index$"
	elif(groupby == 'month'):
		temp['timestamp'] = temp['timestamp'].dt.month
		xlabel = "$month$"
		step = 12
	#elif(groupby == 'year'):
	temp['year'] = news_df_topics['timestamp'].dt.year

	# creamos dataframe fila=topic, columna=timestamp
	d_news_grouped_by_topic = {}
	for index, row in temp.iterrows():
		year      = int(row['year'])
		timestamp = row['timestamp'] + (year-2017)*step
		topic_id  = row['topic']
		if not timestamp in d_news_grouped_by_topic:
			d_news_grouped_by_topic[ timestamp ] = [ 0 for val in range(n_topics) ]

		#if topic_labels!=None:
		#	d_news_grouped_by_topic[ timestamp ][ topic_labels[ topic_id ]  ] += 1
		#else:
		d_news_grouped_by_topic[ timestamp ][ topic_id ] += 1

	# normalizamos por cantidad de noticias por semana
	for key, l_values in d_news_grouped_by_topic.items():
		sum_ = float(sum(l_values))
		d_news_grouped_by_topic[ key ] = [ val/sum_ for val in l_values ]

	# normalized frequency of matrix as dataframe
	data = pd.DataFrame.from_dict(d_news_grouped_by_topic)

	if(format!=None): plt.clf()
	
	fig, ax = plt.subplots()
	
	ax = sns.heatmap(data, ax=ax, annot=False, cmap="Blues", cbar=False)
	#ax.set_xticklabels(temp['timestamp'])

	# set values for colorbar
	cbar = ax.figure.colorbar(ax.collections[0])
	cbar.set_ticks([.0, 0.25, 0.50, 0.75, 1.0])
	cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

	# set xticks when analysis is by month
	if (groupby == 'month'):
		d = {0: 'Jan', 1: 'Feb', 2: 'Mar', 3:'Apr', 4:'May', 5:'Jun', 6:'Jul', 7:'Ago', 8:'Sep', 9:'Oct', 10:'Nov', 11:'Dec'}
		labels = ax.get_xticklabels()
		labels = [ int(l.get_text()) for l in labels ]
		ax.set_xticklabels([ "%s 2018" % d[(m-1) % 12] if (m-1)>=12 else "%s 2017" % d[(m-1) % 12] for m in labels ])

	# set the topic label for the plot
	ylabel = "$Topic\ index$"
	if (topic_labels!=None):
		ax.set_yticklabels([ "$%s$" % val.replace(" ","\ ") for val in topic_labels ])
		ylabel = "$Topic$"

	if (topic_labels!=None):
		if (len(topic_labels)>5):
			rotation=0
		else:
			rotation='vertical'
		#ax.set_xticklabels([ "$%s$" % val.replace(" ","\ ") for val in topic_labels ], rotation=rotation)
		ylabel = "$Topic$"
		plt.yticks(rotation=rotation)
		plt.xticks(rotation=90)
	#plt.xticks(rotation=90)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	if format!= None:
		plt.tight_layout()
		plt.savefig("%stimeline_%s.%s" % (fig_folder, dsname, format))
	else:
		plt.show()
