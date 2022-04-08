download.file(url="http://curl.haxx.se/ca/cacert.pem", destfile="cacert.pem")
library("twitteR")
library("wordcloud")
library("tm")
library("plyr")
library("stringr")

consumer_key = '9dQRbRT4GMOSJUaO8kz2dU9zg'
consumer_secret = 'QzA71tM583X2XvKjxX4OVzhNqYrfYGO8iS8EgDVeXlaHwD1tAT'
access_token = '399702108-yoKDpEZWwesmOCC037MlGuqXrEGE7klqYJLxWoB3'
access_secret = 'jrcBmG5eExilp5i5q73WyO6XG6PagsfwhbLLVR4CIWOHD'
setup_twitter_oauth(consumer_key,consumer_secret,access_token,access_secret)

# 1.2.1 
playstation.tweets = searchTwitter('ps5', lang='en', n=1500)
xbox.tweets = searchTwitter('xbox1', lang='en', n=1500)

# 2.1.1 To transform the tweets into character vectors
playstation.text = laply(playstation.tweets, function(t) t$getText() )
xbox.tweets.text = laply(xbox.tweets, function(t) t$getText() )

# 2.1.3 To remove all non-text characters
playstation = str_replace_all(playstation.text,"[^[:graph:]]", " ")
xbox = str_replace_all(xbox.tweets.text,"[^[:graph:]]", " ")

##### LOADING THE OPINION LEXICON #####

hu.liu.pos = scan('positive-words.txt', what='character', comment.char=';')
hu.liu.neg = scan('negative-words.txt', what='character', comment.char=';')
pos.words = c(hu.liu.pos, 'upgrade')
neg.words = c(hu.liu.neg, 'wtf', 'wait', 'waiting','epicfail', 'mechanical')

##### COPYING THE SENTIMENT SCORE FUNCTION #####
score.sentiment <- function(sentences, pos.words, neg.words, .progress='none')
{
  require(plyr)
  require(stringr)
  
  # we got a vector of sentences. plyr will handle a list
  # or a vector as an "l" for us
  # we want a simple array of scores back, so we use
  # "l" + "a" + "ply" = "laply":
  scores = laply(sentences, function(sentence, pos.words, neg.words) {
    
    # clean up sentences with R's regex-driven global substitute, gsub():
    sentence = gsub('[[:punct:]]', '', sentence)
    sentence = gsub('[[:cntrl:]]', '', sentence)
    sentence = gsub('\\d+', '', sentence)
    # and convert to lower case:
    sentence = tolower(sentence)
    
    # split into words. str_split is in the stringr package
    word.list = str_split(sentence, '\\s+')
    # sometimes a list() is one level of hierarchy too much
    words = unlist(word.list)
    
    # compare our words to the dictionaries of positive & negative terms
    pos.matches = match(words, pos.words)
    neg.matches = match(words, neg.words)
    
    # match() returns the position of the matched term or NA
    # we just want a TRUE/FALSE:
    pos.matches = !is.na(pos.matches)
    neg.matches = !is.na(neg.matches)
    
    # and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():
    score = sum(pos.matches) - sum(neg.matches)
    
    return(score)
  }, pos.words, neg.words, .progress=.progress )
  
  scores.df = data.frame(score=scores, text=sentences)
  return(scores.df)
}
playstation.scores = score.sentiment(playstation, pos.words, neg.words, .progress ='text')
xbox.scores = score.sentiment(xbox, pos.words, neg.words, .progress='text')

##### Creating a Histogram of the Scores #####
par(mfrow =c(1,2),mar=c(2,2,2,2))
hist(playstation.scores$score)
hist(xbox.scores$score)

##### 2.5.4. To calculate the average sentiment score of each phone #####
AvgPlaystationScore = mean(playstation.scores$score)
AvgXboxScore = mean(xbox.scores$score)

AvgPlaystationScore
AvgXboxScore

##### 2.5.7. What analysis can you gather from the sentiment score and histograms? #####
# Based on the histogram and sentiment score, "ps5" has more positive sentiments than "xbox1" on Twitter.

##### Creating the word cloud #####
set.seed(4363)
par(mfrow =c(1,2),mar=c(2,2,2,2))
wordcloud(playstation, min.freq=5, scale=c(3.5, .5), random.order=FALSE, rot.per=0.35, max.words=125, colors='dodgerblue3')
title("PS5 Wordcloud", col.main="grey14")
wordcloud(xbox, min.freq=5, scale=c(3.5, .5), random.order=FALSE, rot.per=0.35, max.words=125, colors='seagreen')
title("Xbox1 Wordcloud", col.main="grey14")