---
layout: post
title:  "A textmining application for the People's Daily newspaper"
date:   2020-06-01 18:32:04 +0200
categories: textmining china peopl's-daily newspaper shiny app
---
What is in the news today? This question can be easily answered if you live in a western country. For people living in China this task can be a mundane job. A small research group consisting of two PhD students and me tackled this task and created a shiny App which uses a textmining tool to analyze the biggest chinese newspaper ([People’s Daily ](paper.people.com)) with descriptive statistics. An abstract of our scientifc report follows a short guide to the Chinese Newspaper Textmining app. For the full report please contact me via mail.

Ever since it’s establishment in 1948 as the Chinese Communist Party’s (CPC) officially
designated publication organ, and especially since it announced the foundation of the
People’s Republic by Mao Zedong on October 1st, 1949, the People’s Daily newspaper
(also known as Renmin Ribao or RMRB) has been an object of the highest interest
for anyone interested in modern China. It is safe to say that it’s front page is that
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
1automate the task of updating the news corpora and economic data, as well as some
descriptive and basic analytical steps used in quantitative text analysis. To make these
results available to non-Chinese speakers, we include a simple translation routine, that
gives the most common word for word translation, even if not the meaning of entire
articles or sentences.

We stress that this will in no way substitute any qualitative reading, knowledge of
Chinese politics, expertise in Chinese language or even text mining of the news in
general. But to be able to quickly develop and test quantitative hypotheses about
Chinese news and its relation to economic data, might be a useful tool on the way to
further research and in-depth text mining

[People's Daily Mining](https://ds-blog.shinyapps.io/Chinese_Newspaper_Textmining/)
