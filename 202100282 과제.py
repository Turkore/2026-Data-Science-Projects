#기본 라이브러리
import pandas as pd 
import re
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import font_manager, rc
from wordcloud import WordCloud
# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Korean
from kiwipiepy import Kiwi

#Download Libraries
style.use('ggplot')
plt.rcParams['font.family'] = "AppleGothic"
plt.rcParams['axes.unicode_minus'] = False
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('universal_tagset')
wc = WordCloud(font_path= 'AppleGothic', width= 800, height= 800, background_color= 'white')

# 분석기 초기화 및 불용어 할당
english_stopwords = set(stopwords.words('english'))
ko_stopwords = ['제', '저', '이', '그', '것', '들', '의', '를', '에', '가', '은', '는', '을']
lemma = WordNetLemmatizer()
r_symbol = re.compile(r'\w+')
sia = SentimentIntensityAnalyzer()
kiwi = Kiwi()

def preprocessing_english(raw_text):
    tokens = nltk.word_tokenize(raw_text)
    analysed_eng = [i.lower() for i in tokens if not i.lower() in english_stopwords and r_symbol.match(i)]
    pos_word = nltk.pos_tag(analysed_eng , tagset= "universal")
    lemmatised_english = [lemma.lemmatize(w, pos={'ADJ':'a', 'VERB':'v', 'ADV':'r'}.get(t, 'n')) for w , t in pos_word]
    return pd.Series(lemmatised_english).value_counts()

def preprocessing_korean(raw_text):
    nouns_kor = kiwi.analyze(raw_text)
    analysed_kor = [token.form for token in nouns_kor[0][0] if token.tag.startswith("N")]
    lemmatised_korean = [w for w in analysed_kor if not w in ko_stopwords and len(w) > 1 and r_symbol.match(w)]
    return pd.Series(lemmatised_korean).value_counts()

def sentiment(raw_text,person_name):
    sentiment = sia.polarity_scores(raw_text)
    sentiment_pd = pd.DataFrame([sentiment], index = [person_name])
    sentiment_pd[['pos', 'neg', 'neu']].T.plot.pie(subplots = True, figsize =(11,6))
    plt.title(f"{person_name}'s Sentiment")
    plt.show()

def wordcloud(data1,data2,title1,title2):
    plt.figure(figsize=(15,7))

    plt.subplot(1,2,1)
    wc1 = wc.generate_from_frequencies(data1.to_dict())
    plt.imshow(wc1, interpolation= 'bilinear')
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1,2,2)
    wc2 = wc.generate_from_frequencies(data2.to_dict())
    plt.imshow(wc2, interpolation= 'bilinear')
    plt.title(title2)
    plt.axis('off')
    plt.show()

def visualisation(data_eng,data_kor,title1, title2):
    fig, ax = plt.subplots(1,2, figsize =(15,6))
    data_eng[0:10].plot(kind='bar', ax=ax[0], color='blue')
    data_kor[0:10].plot(kind='bar', ax=ax[1], color='red')

    ax[0].set_title(title1)
    ax[1].set_title(title2)
    plt.show()

def main():
    with open("/Users/takanomiyayuuki/Desktop/Univ/project/trump2025.txt") as f:
        text = f.read()

    with open("/Users/takanomiyayuuki/Desktop/Univ/project/lee.txt") as f:
        text2 = f.read()
    
    trump_counts = preprocessing_english(text)
    lee_counts = preprocessing_korean(text2)
    sentiment(text, "Trump")
    wordcloud(lee_counts, trump_counts, "Lee's Wordcloud", "Trump's Wordcloud")
    visualisation(trump_counts, lee_counts, "Lee's Top 10", "Trump's Top 10")

if __name__ =='__main__':
    main()