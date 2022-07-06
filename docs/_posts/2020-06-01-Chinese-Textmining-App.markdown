---
layout: post
title:  "A textmining application for the People's Daily newspaper"
date:   2020-06-01 18:32:04 +0200
tags: textmining china peopl's-daily newspaper shiny app
---
What is in the news today? This question can be easily answered if you live in a western country. For people living in China this task can be a mundane job. A small research group consisting of two PhD students and me tackled this task and created a shiny App which uses a textmining tool to analyze the biggest chinese newspaper ([People’s Daily ](paper.people.com)) with descriptive statistics. An abstract of our scientifc report follows a short guide to the Chinese Newspaper Textmining app. For the full report please contact me via mail.

Ever since it’s establishment in 1948 as the Chinese Communist Party’s (CPC) officially
designated publication organ, and especially since it announced the foundation of the
People’s Republic by Mao Zedong on October 1st, 1949, the People’s Daily newspaper
(also known as Renmin Ribao or RMRB) has been an object of the highest interest
for anyone interested in modern China. It is safe to say that its front page is that
of the most widely published newspapers in Chinese and it’s the most widely read.
This amount of attention and it’s clear designation as voice of the CPC have made
it an invaluable source for information on China’s ruling elite’s communication with
the masses. In times of crisis, even tiny changes in placing, formatting or wording are
chosen and interpreted with extreme scrutiny Tan [(1990)](http://www.
jstor.org/stable/2759720). The reason for this is on
the one hand the need for some form of discussion involving the government and the
educated citizenry Kuhn [(2002)](https://www.sup.org/books/title/?id=1845), but also the sensitivity of certain topics, also called
censorship.

In the past, close reading and an intimate knowledge of the Chinese language were
the only tools available to researchers interested in this publication. Even the most
well-read scholars of modern China will have to admit that reading the paper daily
and in it’s entirety, even just the front page, will be a thankless undertaking. Most
articles are official statements and collections of facts about the activites of leaders,
or plain positive messages about some aspect about the nation, also called, by the
CPC themselves, propaganda. The nuances that to detect require years of study are
difficult to check against factual evidence, short of spending hours of reading yourself.
Only rarely are messages communicated as clearly, as back on the founding day of the
People’s Republic.

This presents in our view a very urgent opportunity for automated data mining.
Unlike historic literary corpora like the works of Shakespeare that can be analysed by
generations of scholars, news is by its nature fleeting and often needs to be analysed in
a hurry and theories tested as events unfold. Fortunately, the People’s Daily offers the
whole paper’s content every day for free on its homepage, [People’s Daily ](paper.people.com).
To test our idea for an automated evaluation of People’s Daily articles on their own
and in context of economic data, we proposed the development of an app, that would
automate the task of updating the news corpora and economic data, as well as some
descriptive and basic analytical steps used in quantitative text analysis. To make these
results available to non-Chinese speakers, we include a simple translation routine, that
gives the most common word for word translation, even if not the meaning of entire
articles or sentences.

We stress that this will in no way substitute any qualitative reading, knowledge of
Chinese politics, expertise in Chinese language or even text mining of the news in
general. But to be able to quickly develop and test quantitative hypotheses about
Chinese news and its relation to economic data, might be a useful tool on the way to
further research and in-depth text mining.

Formally, the functionality puts our app in the toolbox of so called *distant reading* approaches in methods of text analysis. The use of which in social sciences in general and economics in particular is growing. As discussed in this [workshop paper](http://ceur-ws.org/Vol-1786/scrivner.pdf), this method is not designed to substitute or render *close reading* obsolete. It actually increases the value of such traditional methods involving a detailed analysis of single texts, by pointing out interesting texts and features in a growing sea of available texts. To think of an overly simple example, compare reading a list of references to reading every single article contained therein. The list does not give the same information, because it abstracts from the detailed texts. But with the list, we can find the articles we are looking for, saving time and leading to new insights, i.e. information we might have missed.

The **translation and transformation** aspect makes the data more useful by offering a translation and transformed version suitable for plotting and analysis. Having a word-for-word translation of the newspaper data allows users to gain a limited impression of the content, allow for quick scanning of the text by users not sufficiently proficient in Chinese, and also allows for streamlined mining. In particular, the free Yandex translation API used here, translates several versions of a Chinese word to the same English correspondence. So this $n-m$ matching, where $n>m$, implies that a user who searches using the English word will get more correct hits than for one particular Chinese word, without necessarily losing the meaning. One of our team members is proficient in Chinese and spot checks have confirmed the quality of the data to be sufficient. Most mistranslated words concern less common names and obscure phrases. For literary allusion and poetic nuance, of course, the translation will necessary and always fail.

All the limits of analyzing language quantitatively also apply in Chinese: Rarely is understanding a text as easy as grasping the meaning of the individual words. For example, when searching for the literal translation of *coronavirus*, recent articles will yield surprisingly low hits. The reason is, that like a lot of Chinese texts, the news uses shorthands or less direct terms, especially for a subject that is politically and emotionally sensitive like this one. Therefore a better search would look for the Chinese equivalent of "outbreak", which in the first months of 2020 is only understood in one way, as a reference to the pandemic.

This does not affect general inquiries, about the most frequent words. In the same way. Censorship and propaganda are more obvious on the aggregate level. It shouldn't surprise anyone to see that the name of president *Xi Jinping*, the *CPC* and other government organs feature highly among the most common words. This is not a mistake, but rather revealing an important aspect of the People's Daily's reporting!

We have gathered around 95000 unique terms over around 15 months. For comparison, the free online dictionary [MDBG.net](https://www.mdbg.net/chinese/dictionary?page=cedict) is based on the CC-CEDICT database which as of 2020-04-06 counts around 120000 entries, so this does not seem entirely unreasonable.

If you want to create your own word frequency plot with our dataset open the app with the followed link and click on the tab *Word Frequencies*. The app might take up to 30 seconds to load and to produce a new plot. So please be a little bit patient!

[People's Daily Mining](https://ds-blog.shinyapps.io/Chinese_Newspaper_Textmining/)
