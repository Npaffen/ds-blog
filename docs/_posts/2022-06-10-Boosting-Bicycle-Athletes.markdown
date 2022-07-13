---
layout: post
title:  "Boosting the bicycle athlete"
math: true
date:   2022-06-12 13:53:16 +0200
tags: strava.com crawler datamining scraping gradient-boosting-models gradient-descent SHAPE-value decision-tree random-forest catboost xgboost lightgbm
---
My recently finished master thesis dealt with the subject of gradient boosting models applied to a dataset scraped from [www.strava.com](strava.com). A scraper written in *C#* was used to collect over 60.000 observations of training exercises from professional bicycle athletes. A diagram of the scraper is added below.

![Diagram of the strava scraper](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/strava_scraper.png?raw=true)
*  $$\textit{Note.}$$Diagram of the working process of the Strava scraper. All user interactions are marked by full line arrows, all automized interactions by the strava scraper are highlighted by coloured dotted arrows. Colours were added just for visual purpose. "*" - signs at the end of a description indicate that this process interacts with strava.com. Therefore a sleep timer after each url call is used to prevent a timeout.*

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

The following table gives an overview of all variables of the Strava dataset. Those variables that have a $$^*$$ were used in the regression models, as explained in later in detail.
![Variable list of the strava dataset](https://raw.githubusercontent.com/Npaffen/ds-blog/99195a58fee048551982e4120d825d7dd66bc808/docs/assets/var_table.png)
*Variable list of the strava dataset*

The measurement $$avg\_power$$ is the actual measurement of the average power provided by the bicycle computer and will be used as one variable to predict the average power of the Strava Dataset.  $$avg\_power\_weig$$ is the adjusted avg. power of the ride where an algorithm from Strava.com corrects possible outliers in the data due to environmental impacts such as terrain, grade, wind and other factors. The variable $$estAvgPower$$ is a guess of the average power measurement from Strava.com if there is no power data supplied by the bicycle computer.  Karetnikov [(2019)](https://research.tue.nl/en/studentTheses/application-of-data-driven-analytics-on-sport-data-from-a-profess) argued that a mean power threshold below 100 is unreasonable and should be skipped.  Therefore we excluded every observation where $$avg\_power$$ or $$average\_power\_combined$$ were lower than 100 watt due to possible negative influence on the prediction models. To maintain as many observations as possible of the strava dataset, we decided to choose those where none of the three average power  measurements showed a value below 100 watt. So that $$avg\_power\_comb$$ was manufactured in the following sense :

$$
\begin{equation}
avg\_power\_comb_j =
\begin{cases}
 estAvgPower_j & \mbox{if} \quad  avg\_power_j \ < 100\\  & \mbox{and} \quad avg\_power\_weig_j < 100 \\ & \mbox{and}\quad estAvgPower_j \geq 100  \\
avg\_power\_weig_j & \mbox{if} \quad estAvgPower_j < 100 \\  & \mbox{and} \quad  avg\_power_j  < 100 \\ &  \mbox{and}\quad avg\_power_j \geq 100 \\
avg\_power_j  &  \mbox{else}
\end{cases}
\label{eq:avg_power_comb}
\end{equation}
$$

$$avg\_power\_comb$$ will be used as a second prediction variable for the average power measurement in a separated model from the $$avg\_power$$ variable. Models that contain the $$avg\_power\_comb$$ variable consist of more observations compared to those that contain the original measurement $$avg\_power$$.

##IRMI

We found that 6880 observations in the Strava dataset contained missing values for either the $$avg\_temperature$$ and/or $$avg\_calories$$ variable(s).
An attempt to deal with missing values is to impute them using variables which were observed or calculated directly from the bicycle computer. To handle this problem we decided on two different strategies. First, the observations which contained $$NA$$ values were dropped, second those values were instead imputed. The imputation was implemented using the IRMI algorithm mentioned first by Templ et al. [(2011)]( https://doi.org/10.1016/j.csda.2011.04.012). The basis for the work of the previous mentioned authors is the IVEWARE algorithm from Raghunathan et al. [(2001)](https://www.researchgate.net/publication/244959137_A_Multivariate_Technique_for_Multiply_Imputing_Missing_Values_Using_a_Sequence_of_Regression_Models) which generates iterative estimates for the missing values using a chain of regression models and picking values from the generated predictive distributions. The IRMI algorithm solves the inability of the IVEWARE algorithm to produce robust results for data including outliers, adds more flexibility by removing the restriction of at least one fully observed variable. In the latter process an user-specified amount of the most important variables for the imputation sequence were chosen. For a formal and detailed explanation about IRMI and IVEWARE please consider both papers mentioned in this paragraph.

To impute variables with $$NA$$ values we chose those 5 variables from the Strava dataset, that have the highest absolute correlation with the variables to be imputed. Therefore the regression models we constructed are presented in  the following two equations:
 \begin{equation}
 \begin{aligned}
 avg\_calories = {mov\_time\_sec} + {distance} + {work\_total} \\  + {elevation} + {max\_speed} + \epsilon
 \end{aligned}
 \label{eq:irmi_reg_1}
 \end{equation}
  \begin{equation}
 \begin{aligned}
  avg\_temperature = {max\_speed}+{elevation}+{distance}  \\ + {distance}+{max\_heartRate} + \epsilon
 \end{aligned}
 \label{eq:irmi_reg_2}
 \end{equation}
