---
layout: post
title:  "Boosting the bicycle athlete"
math: true
date:   2022-06-12 13:53:16 +0200
tags: strava.com crawler datamining scraping gradient-boosting-models gradient-descent SHAPE-value decision-tree random-forest catboost xgboost lightgbm
---
My recently finished master thesis dealt with the subject of gradient boosting models applied to a dataset scraped from [www.strava.com](strava.com). A scraper written in *C#* was used to collect over 60.000 observations of training exercises from professional bicycle athletes. A diagram of the scraper is added below.

![Diagram of the strava scraper](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/strava_scraper.png?raw=true)
*  $$\textit{Note. Diagram of the working process of the Strava scraper. All user interactions are marked by full line arrows, all automized interactions by the strava scraper are highlighted by coloured dotted arrows. Colours were added just for visual purpose. "*" - signs at the end of a description indicate that this process interacts with strava.com. Therefore a sleep timer after each url call is used to prevent a timeout.}$$*

The strava dataset contains training sessions from 185 professional road cyclists from the UCI World Teams list which featured 19 teams in 2020. The strava dataset contains 61840 observations. The Strava dataset contains training sessions from 185 professional road cyclists from the UCI World Teams list which featured 19 teams in 2020. The Strava dataset contains 61840 observations. Table 1 presents an overview of all variables of the Strava dataset. 18 variables were obtained or generated from training activities of the Strava website between 2018-01-01 and 2021-0318. The variables age, height, climber_points, sprinter_points were obtained by data from procyclingstats.com. The two latter variables are the points an athlete achieved for their race results in climbing or sprinting competitions respectively. These variables were then used to create a new variable type. Each observation of the Strava dataset was matched with the aggregated point score from the first day of the year to the activity day of the year for each athlete gained from climbing and sprinting races respectively. Let $$d_{l}$$ be defined as the date of an observation $$i$$ of the Strava dataset of athlete $k, \mathrm{D}$ is the date of the observation for which the categorization type was determined and $$D \geq d$$ is valid. Then, we can define the type of each athlete $$k$$ at each training activity $$i_{D}$$ by the following rule:

$$
\begin{gather}
\scriptsize
type_{i_D, k} =
\begin{cases}
climber  & \mbox{if} \quad \sum_{i_{d}}^{i_{D}}climber\_points_{i_d, k} >  \biggl\{ \sum_{i_{d}}^{i_{D}}sprinter\_points_{i_d, k} \biggr\}*1.25 \\
& \mbox{and} \quad  \biggl\{\sum_{i_{d}}^{i_{D}}climber\_points_{i_d, k} -  \sum_{i_{d}}^{i_{D}}sprinter\_points_{i_d, k} \biggr\} \geq 30  \\
sprinter & \mbox{if} \quad  \sum_{i_{d}}^{i_{D}}sprinter\_points_{i_d, k} >  \biggl\{ \sum_{i_{d}}^{i_{D}}climber\_points_{i_d, k} \biggr\}*1.25   \\
 & \mbox{and} \quad  \biggl\{\sum_{i_{d}}^{i_{D}}sprinter\_points_{i_d, k} -  \sum_{i_{d}}^{i_{D}}climber\_points_{i_d, k} \biggr\} \geq 30 \\
mixed   &  \mbox{else}
\end{cases}
\label{eq:type_rule}
\end{gather}
$$
