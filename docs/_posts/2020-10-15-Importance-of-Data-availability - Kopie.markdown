---
layout: post
title:  "Importance of data avaibility - Lessons from a replication study"
math: true
date:   2020-10-15 16:24:33 +0200
tags: Brunello replication-study SHARE SHARE-dataset data-avaibility

---
For the university course *Causality and Programme Evaluation* I recently did a replication study of Brunello et al.[2009](https://www.jstor.org/stable/20485330). This study showed how extremly time-consuming due application procedures are to get data for a replication study. Sometimes, it is even impossible to get the data in a reasonable amount of time and without the help of a academic institution. The result of this lack of data avaibility for scientific research are biased results in the replication process or the inability to check the emperical output of a scientific paper. This blog post will compare my replicated results with those from the original paper.

From the introduction of my term paper "Males Catch up"
>The replication study focused on the summarization and replicated results of Brunello et al.[2009](https://www.jstor.org/stable/20485330). The main idea of the paper was to show if extending >years of compulsory schooling has an effect on the distribution of wages. Further findings might be that compulsory school reforms significantly affect educational attainment, which holded in >this replication study for almost none quantile, since the given data did not help to show the possible effect of the instrument years of compulsory schooling. The replication study  was >limited to the data of the SHARE data set, compared to a mixed dataset consisting of a "data[set] drawn from the 8th wave of the European Community Household Panel (ECHP) for the year 2001, >the first wave of the Survey on Household Health, Ageing and Retirement in Europe, or SHARE, for the year 2004, and the waves 1993 to 2002 of the International Social Survey Program (ISSP)" >Brunello et al.[2009](https://www.jstor.org/stable/20485330). Due to the latter difference, replicated results differ heavily from those in the original paper.

The main idea of Brunello et al.[2009](https://www.jstor.org/stable/20485330) was to use years of compulsory schooling (*ycomp*) to identify the impact of an additional year of schooling on wages with respect to the gender of the individual. The author of the orignal paper used the Chesher's approach from Chesher [(2010)](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA12223)  to propose an exactly identified triangular model.
To solve the latter and thereby generate a consistent estimator, @brunello propose a variable $z$ that is corellated with schooling but orthogonal to individual ability conditional on schooling and orthogonal to the endogenous variable of (8). The instrumental variable in this model is years of compulsory schooling $$ycomp$$. Taken this into account the (exactly identified triangular) model can be expressed as :

$$
\begin{align}
\ln(w)&=\beta s+s(\lambda a+\phi u)+\gamma_{w} X+a+u \\
s&=\gamma_{s} X+\pi z+\xi a
\end{align}
$$

with $$\xi=(\lambda+\kappa) / \theta$$ Let $$\tau_{a}=G_{a}\left(a_{\tau_{a}}\right)$$ and $$\tau_{u}=G_{u}\left(u_{\tau_{u}}\right)$$, where $$a_{\tau_{a}}$$ and $$u_{\tau_{u}}$$ are the $$\tau$$ - quantiles of the distributions of $$a$$ and $$u$$, respectively. Additionally define f$$Q_{w}\left(\tau_{u}\mid s,X,z\right)$$ and $$Q_{s}\left(\tau_{a}\mid X,z\right)$$ as the conditional quantile functions corresponding to log wages and years of education. To achieve the recursive conditioning model one needs to compute the control variates first. Step one is to estimate the conditional quantile functions of schooling $$s$$  and afterwards subtract the estimated values of the specific quantile from years of schooling. Considering (8) again and the fact that the model is exactly identified one only remains  with the value of ability at the specific quantile tau. Formally :

$$
\begin{align}a\left(\tau_{a}\right)=s-\bar{Q}_{s}\left(\tau_{a}\mid X,z\right)
\end{align}
$$

Afterwards one adjusts the conditional quantile functions of $$ln(w)$$ with the control variate of (9) so that the residuals, orthogonal to ability, of the estimated conditional quantile regression of $$ln(w)$$ yields to $$u(\tau_u)$$ of the following regression equation :

$$
\begin{align}
\tilde{Q}_w[\tau_u|X, s, a(\tau_a)] = \beta s + s(\lambda a(\tau_a) + \phi u(\tau_u)) + \gamma_w X + G_{a}^{-1}\left(\tau_{a}\right) + G_{u}^{-1}\left(\tau_{u}\right)
\end{align}
$$

Now one can construct the parameter $\Pi(\tau_a,\tau_u)$ which is a matrix with the following structure :

  $$
  \begin{align} \Pi\left(\tau_{a},
 \tau_{u}\right)=\beta+\lambda G_{a}^{-1}\left(\tau_{a}\right)+\phi
 G_{u}^{-1}\left(\tau_{u}\right)
 \end{align}
 $$

Due to recursive conditioning $$Q_{s}\left(\tau_{a}\mid X,z\right)$$ on $$Q_{w}\left[\tau_{u}\mid X,z\right)$$ one yields to the following model :

$$
\begin{align} Q_{w}\left[\tau_{u}
 \mid Q_s\left(\tau_{a} \mid X, z\right), X, z\right]&=Q_s\left(\tau_{a}
 \mid X, z\right) \Pi\left(\tau_{a}, \tau_{u}\right)+\gamma_{w}
 X+G_{a}^{-1}\left(\tau_{a}\right)+G_{u}^{-1}\left(\tau_{u}\right)&  \\
 Q_{s}\left(\tau_{a} \mid X, z\right)&=\gamma_{s} X+\pi z+\xi
 G_{a}^{-1}\left(\tau_{a}\right) &
 \end{align}
 $$

A two stage fit of the latter models then gives us the coefficient of $$Q_s\left(\tau_{a}\mid X, z\right)$$ After we plug this into $$\Pi\left(\tau_{a},\tau_{u}\right)$$ for the coefficient $$\beta$$. We will repeat this step for each quantile $$\tau (0.1, 0.3, 0.5, 0.9)$$ but always altering only the quantiles of either $$\tau_{u}$$ of $$Q_{w}\left[\tau_{u}\mid Q_s\left(\tau_{a} \mid X, z\right), X, z\right]$$  or $$\tau_{a}$$ of the parameter $$Q_s\left(\tau_{a}\mid X, z\right) \Pi\left(\tau_{a}, \tau_{u}\right)$$ . Thereby the study observes in the first case the effect of  how a specific quantile $$\tau_a$$ of $$Q_s\left(\tau_{a}\mid X, z\right)\Pi\left(\tau_{a},\tau_{u}\right)$$ interacts with the entire distribution of the log hourly earnings , while the latter measure the effect of the different quantiles of the ability distribution on the fixed $$\tau_u$$ of the endogenous variable $$ln(w)$$. Integrating the key parameter $$\Pi\left(\tau_{a},\tau_{u}\right)$$ with respect to $$\tau_a$$ results in mean quantile treatment effects. The latter gives an overview of how an individual with average abilites is rewarded for educational attainment in the different quantiles of the labour market luck distribution.

Less formally the triagular model can be explained as the following: Assume that the earnings $$w$$ of an individual are correlated with their educational level. Let this educational level in this model be defined as years of schooling $$s$$. Therefore we need an instrument which correlates with the educational level measurement but not with the earnings $$w$$.

Lets compare some of the results from the original paper and my replication study. The first result is a simple estimation of the the average impact of a compulsory school reforms of several european countries. The first graph is always from the original paper the second one from my replication study.

![The Effect of School Reforms on Educational Attainment](https://github.com/Npaffen/ds-blog/blob/main/docs/_site/assets/brunello_resid.png?raw=true)

 *Note. The OLS gender-specific regressions included a constant, country dummies, $$q$$, $$q^2$$ and their
interaction with country dummies, survey dummies, age, age squared, the lagged country specific
unemployment rate and GDP per capita, the country and gender specific labour force participation
rate at the estimated time of labour market entry, the country specific GDP per head and
unemployment rate at the age affected by the country specific reform.*

The authors of the original paper showed that their model which estimates the years of schooling, showed an average increase of the estimate of 0.3 years of schooling of each pupil cross-country wise, when a compulsory schooling reform was introduced. This effect was measured through an increase of the mean residual value which has to be an external effect, e.g. a school reform, since the authors claimed to have an "exactly identified triangular model".


![The Effect of School Reforms on Educational Attainment](https://github.com/Npaffen/ds-blog/blob/main/docs/_site/assets/brunello_resid_repli.png?raw=true)

*Note. The OLS gender-specififc regressions included a constant, country dummies, $$q$$, $$q^2$$
and their interactions with country dummies and the GDP per head at the age when
the pupil would have finished compulsory schooling.*

The opposite can be observed inspecting the change over time of the resiudal values in my replication study. We observe residual values around zero over time and measure even a decrease of the mean residual values after the shock due to the school reform.



Most of the coefficients, of table 1 of my replication study, are insignificant and/or unlikely high. Examples are the 10th and 90th quantile for males of $$\tau_{a}$$ when $$\tau_{u}$$ and the 30th quantile for females. In comparison, in table 2, Brunello et al.[2009](https://www.jstor.org/stable/20485330) finds that the estimated increase in earnings for males at the lowest quantile are $$7.48 \%$$ compared to roughly $$1800 \%$$ in the result of my replication study and stay at a very high percantage level for mostly around $$700 \%$$ to $$800 \%$$ percent. The page for the females looks quite the opposite. It seems that males have to catch up extremly since many of the results from the bottom to the highest quantile of ability show some negative coefficients. But there are some mentionable positive results for the latter gender, too. The findings would expect the estimated returns of education of females which find themselves in the 30th quantile of the ability distribution and the highest, quantile of the labour market, luck distribution of roughly $$5000 \%$$. If the latter would be the case we would expect much more women from the lower ability distribution to earn a fortune after leaving school with an extra year of education while hitting the jackpot in the "labour market luck lottery" aka presenting an indivdiual in the $90 \mathrm{th}$ percentile of the labor market fortune distribution. In a more realistic world presented by the results of Brunello et al.[2009](https://www.jstor.org/stable/20485330) females are thoses to catch up at the fight for wages. The authors found that especially women from the lowest percentile benefit from an extra year of compulsory schooling which is expressed as returns for the surplus education of roughly $$10 \%$$.

![Quantile Effects When Education is Treated as Exogenous](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/Table_4.png?raw=true)
*Table 1 $$\textit{Note.}$$
Each regression included a constant, country dummies, $$q$$, $$q^2$$ and their interactions with country dummies, age, age squared, and the GDP per head at the age when the pupil would have finished compulsory
schooling. $$\tau$$ denotes the quantile of the distribution of wages. Three stars, two stars and one star for statistically significant coefficients at the 1%, 5% and 10% confidence level. Bootstrapped standard errors are shown in parentheses.*

![Quantile Effects When Education is Treated as Exogenous](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/Returns_brunello.png?raw=true)
*Table 2 $$\textit{Note.}$$
Each regression included a constant, country dummies, $$q$$, $$q^2$$ and their interactions with country dummies, age, age squared, the lagged country specific unemployment rate and GDP
per capita, the country and gender specific labour force participatio rate at the estimated time of labour market entry, the country specific GDP per head and unemployment rate at the age affected by the country specific reform. Details on these coefficients are available from the authors upon request. Three stars, two stars and one star for statistically significant coefficients at the 1%, 5% and 10% confidence level. Robust standard errors are shown in parentheses.*

My replication study was far away from actually replicating the results. In my case  I had to blame the data. Overall, blaming the data is sometimes just a bad excuse to actually blame the authors because they did not use all methods from the statistical tool box to analyze the data. The approach of Brunello et al.[2009](https://www.jstor.org/stable/20485330) of a quantile regression model added a new tool to my statistical tool box, so that replicating the study was a good training method wise. I asked the authors of Brunello et al.[2009](https://www.jstor.org/stable/20485330) for the data to replicate the study but never recieved an answer. Sharing data for scientific purposes is still a big problem today. An example of the impact on the results was presented in this blog post!
