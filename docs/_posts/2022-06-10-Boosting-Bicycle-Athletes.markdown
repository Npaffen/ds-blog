---
layout: post
title:  "Boosting the bicycle athlete"
math: true
date:   2022-06-12 13:53:16 +0200
tags: strava.com crawler datamining scraping gradient-boosting-models gradient-descent SHAPE-value decision-tree random-forest catboost xgboost lightgbm
---
My recently finished master thesis dealt with the subject of gradient boosting models applied to a dataset scraped from [www.strava.com](strava.com). A scraper written in *C#* was used to collect over 60.000 observations of training exercises from professional bicycle athletes. A diagram of the scraper can be seen in figure 1.

![Diagram of the strava scraper](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/strava_scraper.png?raw=true)
*  $$\textit{Note.}$$Figure 1 : Diagram of the working process of the strava scraper. All user interactions are marked by full line arrows, all automized interactions by the strava scraper are highlighted by coloured dotted arrows. Colours were added just for visual purpose. "*" - signs at the end of a description indicate that this process interacts with strava.com. Therefore a sleep timer after each url call is used to prevent a timeout.*

The strava dataset contains training sessions from 185 professional road cyclists from the UCI World Teams list which featured 19 teams in 2020. The strava dataset contains 61840 observations. The strava dataset contains training sessions from 185 professional road cyclists from the UCI World Teams list which featured 19 teams in 2020. The strava dataset contains 61840 observations. Table 1 presents an overview of all variables of the strava dataset. 18 variables were obtained or generated from training activities of the strava website between 2018-01-01 and 2021-0318. The variables age, height, climber_points, sprinter_points were obtained by data from procyclingstats.com. The two latter variables are the points an athlete achieved for their race results in climbing or sprinting competitions respectively. These variables were then used to create a new variable type. Each observation of the strava dataset was matched with the aggregated point score from the first day of the year to the activity day of the year for each athlete gained from climbing and sprinting races respectively. Let $$d_{l}$$ be defined as the date of an observation $$i$$ of the strava dataset of athlete $$k, \mathrm{D}$$ is the date of the observation for which the categorization type was determined and $$D \geq d$$ is valid. Then, we can define the type of each athlete $$k$$ at each training activity $$i_{D}$$ by the following rule:

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

The following table gives an overview of all variables of the strava dataset. Those variables that have a $$^*$$ were used in the regression models, as explained in later in detail.
![Variable list of the strava dataset](https://raw.githubusercontent.com/Npaffen/ds-blog/99195a58fee048551982e4120d825d7dd66bc808/docs/assets/var_table.png)
*Variable list of the strava dataset*

The measurement $$avg\_power$$ is the actual measurement of the average power provided by the bicycle computer and will be used as one variable to predict the average power of the strava Dataset.  $$avg\_power\_weig$$ is the adjusted avg. power of the ride where an algorithm from strava.com corrects possible outliers in the data due to environmental impacts such as terrain, grade, wind and other factors. The variable $$estAvgPower$$ is a guess of the average power measurement from strava.com if there is no power data supplied by the bicycle computer.  Karetnikov [(2019)](https://research.tue.nl/en/studentTheses/application-of-data-driven-analytics-on-sport-data-from-a-profess) argued that a mean power threshold below 100 is unreasonable and should be skipped.  Therefore we excluded every observation where $$avg\_power$$ or $$average\_power\_combined$$ were lower than 100 watt due to possible negative influence on the prediction models. To maintain as many observations as possible of the strava dataset, we decided to choose those where none of the three average power  measurements showed a value below 100 watt. So that $$avg\_power\_comb$$ was manufactured in the following sense :

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

## IRMI

We found that 6880 observations in the strava dataset contained missing values for either the $$avg\_temperature$$ and/or $$avg\_calories$$ variable(s).
An attempt to deal with missing values is to impute them using variables which were observed or calculated directly from the bicycle computer. To handle this problem we decided on two different strategies. First, the observations which contained $$NA$$ values were dropped, second those values were instead imputed. The imputation was implemented using the IRMI algorithm mentioned first by Templ et al. [(2011)]( https://doi.org/10.1016/j.csda.2011.04.012). The basis for the work of the previous mentioned authors is the IVEWARE algorithm from Raghunathan et al. [(2001)](https://www.researchgate.net/publication/244959137_A_Multivariate_Technique_for_Multiply_Imputing_Missing_Values_Using_a_Sequence_of_Regression_Models) which generates iterative estimates for the missing values using a chain of regression models and picking values from the generated predictive distributions. The IRMI algorithm solves the inability of the IVEWARE algorithm to produce robust results for data including outliers, adds more flexibility by removing the restriction of at least one fully observed variable. In the latter process an user-specified amount of the most important variables for the imputation sequence were chosen. For a formal and detailed explanation about IRMI and IVEWARE please consider both papers mentioned in this paragraph.

To impute variables with $$NA$$ values we chose those 5 variables from the strava dataset, that have the highest absolute correlation with the variables to be imputed. Therefore the regression models we constructed are presented in  the following two equations:

$$
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
$$

Figure 2 shows the Pearson correlation coefficient in a heat map. Tiles which are more reddish implicate a high positive correlation, white tiles implicate that the two variables are uncorrelated while more bluish tiles mark a strong negative correlation. The variables we want to predict are $$avg\_power$$ and $$UCI\_points\_weekly$$. The variables with the highest correlation for the $$avg\_power$$ were $$work\_total (0.44)$$, $$avg\_calories (0.4)$$ and $$avg\_cadence (0.37)$$. For $$UCI\_points\_weekly$$ we observed a slightly negative correlation with $$type\_mixed$$ and a small positive correlation with $$type\_sprinter$$. Therefore we observed that those athletes who focus more on sprint races or tournaments with many sprint sections seemed to achieve higher UCI points, while athletes of type mixed tended to achieve worse results. The overall low correlation with other variables indicated that a prediction of $$UCI\_points\_weekly$$ with the strava dataset might not achieve good results.

# Methods

This section will quickly summarize which methods and models were used to analyze the dataset. A quick introduction into regression trees follows a diagramm focused approach to explain gradient boosting algorithms. Afterwards I will present a short 101 on  Bayesian HPO and the SHAP approach which is used to shed light on the inside of black-box models.

## Decison Trees
Decision trees are the most basic tree-based models which are part of the non-parametric algorithms class. The latter means that we do not estimate a direct influence of a variable on the model. The algorithm divides a multi-dimensional feature space into unique sets, so called nodes. Generally, we distinguish between two decision tree tasks, classification and regression trees. Since our $$y$$, the variable of interest, $$avg\_power$$, $$avg\_power\_weekly$$ and $$UCI\_weekly\_points$$ are all continuous variables, we will just focus on regression trees. Each split in a regression tree is ruled by the decision of a variable threshold. At the beginning, there is a root node, which contains the entire  dataset and implies the first split into subsets. Each following node from a split can be either a decision node or a terminal node. The further term is used for nodes that will lead to a further split of the dataset. If no further split happens at one node, it is called a terminal node, leaf node, leaf. To decide which variable $$x$$ and which threshold $$j$$ of this variable $$x$$ should be chosen to split the dataset, one uses the sum of squared residuals to evaluate the possible splitting decision  for a regression trees at a node $$t$$.

$$
\begin{equation}
SS_{t} = \sum_i^{n_{1}} (y_1 - \bar{y}_1)^2 + \sum_i^{n_{2}} (y_2 - \bar{y}_2)^2
\end{equation}
$$

with $$y_1=(y_i|x_i>=j)$$, $$y_2=(y_i|x_i>=j)$$. We calculate the $$SS_t$$ for each variable $$x$$ and each threshold $$j$$ of each variable $$x$$ of the dataset. Then we chose the variable-threshold combination that generates the lowest $$SS_t$$ as our splitting criterion.

## Random Forests
Estimating only one decision tree might give us a to narrow solution for the prediction of our target variable. Breiman [(2001)](https://doi.org/10.1023/a:1010933404324) showed that generating several uncorrelated decision trees , in terms of their prediction error, should give on average a better model to predict from than a model from a single tree. This is due to the convergence of the error of the aggregated random forest model. So with an increased number of unique decision trees, the error of the random forest model should converge to the mean average error. To ensure that we aggregate a model of decision trees that have a low correlation between each tree, two methods were used. With bagging, a dataset of $$n$$ observations is drawn with replacement from the dataset. Decision trees are very sensitive to (small) changes in the dataset, since decision rules are based on singular values of a variable. If the value of a decision rules is missing, e.g. due to bagging, a new tree structure is possible, since all subsets after this decision node were affected. The other method is feature subsampling. If we only choose some features of the dataset instead of all and vary those chosen features with each new decision tree, the possibility to end up with an decision tree that is correlated to some other tree from the dataset should be low, and thereby the correlation of the prediction error should be low. To predict the value of the target variable of a new observation, we collect the classification result of each tree in the random forest and choose the class which was predicted by the majority of trees.

![Random Forest](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/random_forest.png?raw=true)
*Figure 3 :Diagram of a random forest prediction example. A new observation is shown to the model and each tree gives its prediction on the target variable, here $$type$$, of the strava dataset. The figure is just for clarification of the concept and does not necessarily represent a possible outcome of a random forest model.*

Figure 3 gives an example of a random forest prediction. The latter prediction of the target variable $$type$$ would be $$sprinter$$, Since 3 out of 4 trees would predict that the new observation would be of type $$sprinter$$f

## Gradient Boosting

Mason [(1999)](https://www.researchgate.net/publication/221618845_Boosting_Algorithms_as_Gradient_Descent) proposed an boosting algorithm based on gradient descent. Gradient Boosting Models generate multiple tree models and add them together to generate a single ensemble model.  The Gradient Boosting machine (GBM) learns with each iteration which splits improved the ensemble model trough a regularized learning objective. Consider some input data $$\{(x_i,y_i)\}^n_{i=1}$$ and a differential loss function $$L(y,F(x))$$. $$F(x)$$ is a model to predict $$\hat{y}$$. A loss function is an evaluation metric of a model $$F(x)$$ and our target variable $$y$$. As an example we will use
$$
 \begin{equation}
 L(y,F(x)) = \dfrac{1}{2}(y - \hat{y})^2
 \label{eq:reg_loss}
 \end{equation}
 $$
 as our loss function. Where the $$F(x)$$ is a model to predict $$\hat{y}$$.

 ![GBM](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/gradient_boosting.png?raw=true)
 *Figure 4: The diagram shows exemplary how GBM calculates the predictions.*

 Figure 4 shows a visualization of this process. We decided to not include the GBM method in the results section since XGBoost  and Lightgbm are direct successors of this technique and are likely to outperform the GBM method. We still included this method to give a good introduction into the following gradient-based tree-ensemble methods.

## XGBoost
 ![XGBoost](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/XGBoost-1.png?raw=true)
 *Figure 5: The diagram shows exemplary how XGBoost initiates the algorithm, builds trees, calculates the weights and updates the model.*

Figure 5 shows a visualization of the XGBoost algorithm. The XGBoost algorithm sets the initial loss $$\mathcal{L}^{(0)} = l(y_i,\hat{y}_i) = 0.5$$.
Again we consider the variable $$avg\_power\_comb$$ as our target variable. So we calculate the initial prediction value from our target variable, obtain the pseudo residuals as described before, and use these pseudo residuals as the dataset for the tree structure $$q$$. Let the 3 splitting rules be $$intensity>0.69$$, $$avg\_speed>35.9$$ and $$type = mixed$$. Then calculate the output value at each leaf node and add the weighted results of this process to each observation of the target variable of each pseudo residual respectively.

## LightGBM

Ke et al. [(2017)](. https://doi.org/10.5555/3294996.3295074) proposed a gradient boosting algorithm called LightGBM (LGBM), which supports a way to differ between instances with small and large absolute gradient statistics. They called this method $$\textit{Gradient-based One-Side Sampling}$$ (GOSS) and argued that those instances with a small gradient also show a low training error and should therefore receive less attention. GOSS sorts the instances with respect to their absolute value of their gradients in descending order and chooses the top   $$l \times 100\%$$ instances. From the other $$1-t$$ share of instances, GOSS randomly samples $$s \times 100\%$$ instances. The latter samples are multiplied with a small weight $$\dfrac{1-l}{s}$$ when the loss function is evaluated.

The following explains the splitting decision for trees in LGBM. $$\tilde{v}_{j}(p)$$ is the estimated variance gain when we split a feature $$k$$ at point $$p$$. As explained before, LGBM differs between instances with large and small absolute gradient statistics. So that $$C_l = \left\{x_{i} \in C: x_{i k} \leq p\right\}$$, $$C_{r}=\left\{x_{i} \in C:x_{ik}>p\right\}$$. The feature values smaller than or equal to the threshold $$p$$ of those instances with large absolute gradient statistics would be split to the left child node and those exceeding the threshold $$p$$ would be split to the right child node. The same definition holds for $$D_l$$ and $$D_r$$, with the difference that $$D$$ represents randomly sampled instances from those with already low absolute gradient statistics. Formally, LGBM estimates $$\tilde{v}_k(p_k^*)$$, because we train with a dataset of instances that is smaller than the dataset of all possible instances, such that
$$
\begin{equation}
\tilde{v}_{j}(p)=\frac{1}{n}\left(\frac{\left(\sum_{x_{i} \in C_{l}} g_{i}+\dfrac{1-l}{s} \sum_{x_{i} \in D_{l}} g_{i}\right)^{2}}{n_{l}^{k}(d)}+\frac{\left(\sum_{x_{i} \in C_{r}} g_{i}+\dfrac{1-l}{s} \sum_{x_{i} \in D_{r}} g_{i}\right)^{2}}{n_{r}^{k}(pd)}\right).
\label{eq:lgbm_split}
\end{equation}
$$

## Catboost
Prokhorenkova et al. [(2019)](https://arxiv.org/abs/1706.09516) proposed an enhanced gradient boosting algorithm called CatBoost. The latter is an abbreviation for categorical boosting. They claim to be the first, who found a solution to the problem of target leakage in gradient boosting. The latter implies that due to the usage of the same training observations, for the calculation of the gradient statistics and models used to minimize those gradients, a prediction shift is produced. They also found a similar problem with the encoding of categorical variables which will be explained at the end of this section.

 CatBoost solves this by a technique called Ordered Boosting.

Let $$I$$ be the amount of trees a model is learned with. Unshifted residuals $$r^{I-1}(\mathbf{x_k},y_k)$$ can only obtained if we exclude observation $$\mathbf{x}_k$$ in the training process of  $$F^{I-1}$$. The authors argue that the training process would become impossible because, since all residuals need to be unshifted, we cannot use any observations for the training process of $$F^{I-1}$$. Prokhorenkova et al. [(2019)](https://arxiv.org/abs/1706.09516) solve this by a set of models which does not include any information of the predicted observation. Therefore, one uses a permutation $$\sigma$$ of $$\mathcal{D}$$ and learns $$n$$ unique models $$M_1,\ldots,M_n$$. Each model $$M_i$$ uses only observations in the permutation $$\sigma$$ between the first and the $$i$$-th observation. The residual of the $$j$$-th observation in the permutation $$\sigma$$ uses the model $$M_{j-1}$$. CatBoost combines the latter method with gradient boosting with symmetrical decision trees to obtain unbiased residuals $$R(\mathbf{x}_k,y_k)$$ and thereby an unbiased combined model $$F$$.

Figure 6 shows how the ordered boosting algorithm works to create the combined model $$F$$. Since the calculation of the residuals and the iterative manner of gradient boosting techniques were explained in detail for GBM and XGBoost, figure 6 focuses more on the unique differences to other gradient boosting techniques.

![CatBoost](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/Ordered_Boosting.png?raw=true)
 *Figure 6: The diagram shows how the orderd boosting algorithm in CatBoost works.*

 Prokhorenkova et al. [(2019)](https://arxiv.org/abs/1706.09516) claim that the prediction shift mentioned before applies in the same way when computing target statistics for a categorical feature. To avoid this, they propose a sub-sampling so that $\mathcal{D}_{k}\subset\mathcal{D}_{\left\{\mathbf{x}_{k}\right\}}$ is used to calculate the target statistics for $x_k$, therefore excluding $x_k$ from the process. Let $p$ be some prior to smooth the estimate $\mathbb{E}(y|x^i=x^i_k)$ for categories of a feature with a low proportion compared to the other categories of that feature, then :
\begin{equation}
\hat{x}_{k}^{i}=\frac{\sum_{\mathbf{x}_{j} \in \mathcal{D}_{k}} \mathbb{1}_{\left\{x_{j}^{i}=x_{k}^{i}\right\}} \cdot y_{j}+a p}{\sum_{\mathbf{x}_{j} \in \mathcal{D}_{k}} \mathbb{1}_{\left\{x_{j}^{i}=x_{k}^{i}\right\}}+a}
\label{target_stat_cat}
\end{equation}


 Prokhorenkova et al. [(2019)](https://arxiv.org/abs/1706.09516) then introduced a technique called ordered targeting statistic. Comparable to the online learning algorithm, we feed the model sequentially, using a random permutation $$\sigma$$ of the training dataset. Thereby, the model observes in each iteration another new training observation of the permuted training dataset, but all other observations from past iterations will be used to compute the target statistic as well.

 ![target encoding](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/ordered_target_statistics.png?raw=truee)
 *Table 6: The diagram shows how the orderd boosting algorithm in CatBoost works.*

 Table 6 shows a less formal explanation of ordered target statistics. Consider each row of $$\textit{target}$$ and $$\textit{type}$$ as single observations of the exemplary dataset. The derivation of the value in column $$\textit{encoded target}$$ is explained in the column $$\textit{explanation}$$. The column $$\textit{history}$$ describes the 'observed history' of the ordered target statistics calculation. Such that, for the value of row with $$history = 3$$, we have $$target = 1$$ and $$type = mixed$$. No 'observed history' exists with $$target = 1$$, and $$type = mixed$$, so $$\sum_{\mathbf{x}_{j} \in \mathcal{D}_{k}} \mathbb{1}_{\left\{x_{j}^{i}=x_{k}^{i}\right\}} \cdot y_{j}+a p$$ in equation \eqref{target_stat_cat} is still 0, but $$\sum_{\mathbf{x}_{j} \in \mathcal{D}_{k}} \mathbb{1}_{\left\{x_{j}^{i}=x_{k}^{i}\right\}}$$ is now 1 since the sum of the denominator is no conditioned on the target value $$y_j$$. The latter is important, since Prokhorenkova et al. [(2019)](https://arxiv.org/abs/1706.09516) argue that a desired property of the target statistics feature calculation and the learning process of the model is the effective usage of all training data. Still, the oblivious counter of the fraction satisfies the requirement to counter the prediction shift. The authors called this bias target leakage, since recognizing the actual target value of the observation for which the $$\textit{encoded target}$$ calculation is performed, leaks information about the distribution of the target to the encoding process. Since this learned distribution might differ from the distribution of the target value in the test dataset, a biased prediction due to overfitting is possible.

## Bayesian HPO

To optimize the tree ensemble models we can optimize their hyperparameters. This subsection will introduce the Bayesian HPO method and will discuss which hyperparameters are possible and useful for each tree ensemble technique.
Tree ensemble algorithm such as the gradient boosting tree implementations of XGBoost, Lightbm and Catboost have several hyperparameter which handle:

$$
\begin{itemize}
\item \textbf{learning rate} - the contribution of each (tree) model, often a factor betwen 0 and 1
\item \textbf{mtry}  - defines how many variables will be used for each (tree) model
\item \textbf{sample size} - defines the number or share of observations of the training dataset to be used in each (tree) model
\item \textbf{tree depth} - the maximum amount of levels each tree can generate
\item \textbf{trees} - the number of trees to build
\item \textbf{minimal node size} - a minimum number of observations in a node needed to accept a new split at a decision node of a tree
\item \textbf {loss reduction} - minimum loss reduction needed to accept a split at a decision node of a tree
\item \textbf{iteration stop} - the number of trees without improvement before stopping the algorithm
\end{itemize}
$$

The aim of HPO is to find a set of hyperparameter values for a (machine learning) method which minimizes the chosen loss function to evaluate the quality of the model on the validation dataset. Let the chosen loss function be some function $$f(x)$$ and $$x^*$$ the set of optimal hyperparameter values, then HPO can be stated as :
$$
\begin{equation}
x^* = \underset{x \in \mathcal{X}}{\arg \min }f(x)
\label{eq:HPO}
\end{equation}
$$

Other HPO methods such as manual search, grid search or random search suffer from the problem that they cannot evaluate the chosen set of hyperparameter values after each iteration. This problem is solved by Bayesian HPO which uses an acquisition function as a prior for the hyperparameter set to evaluate next. Consider an objective function such as equation \eqref{eq:reglu_obj} from XGBoost and a dataset with $$n$$ observations and $$d$$ features, which are the hyperparameters of our objective function, so that $$x_i = (x_{i1},\ldots,x_{id})^T$$ and $$y_i = y_i(x)$$ Either one initialize the optimization process with $$t$$ hyperparameter sets with $$t\in n$$ or the algorithm randomly samples $$t$$ of such sets w.r.t. the range of each hyperparameter.  $$D_0 = {x_1,\ldots,x_n}$$ stores the hyperparameter sets for $$n$$  trials in a $$n \times d$$ matrix $$X$$. Normalization of $$x_i$$ achieves $$x_i \in [0, 1]^d$$. Afterwards, each set $$i$$ is used with the objective function to obtain a performance metric as the $$i$$-th observation of the dependent variable $$y_i$$. Those outputs are stored in the $$n \times 1$$ vector $$Y = y(X) = (y_1,\ldots,y_n)^T$$. Instead of continuing to evaluate the objective function for further hyperparameter sets, Bayesian HPO uses a surrogate model to estimate the function to be optimized. The model used in this implementation is the Gaussian Process (GP) model:

$$
\begin{equation}
y\left(x_{i}\right)=\mu+z\left(x_{i}\right) ; \quad i=1, \ldots, n
\label{eq:GP_model}
\end{equation}
$$
where $$\mu$$ is the overall mean and $$z(x_i)$$ defines the GP with $$E[z(x_i)] = 0$$, $$Var[z(x_i)]=\sigma^2$$ and $$Cov(z(x_i),z(x_j))= \sigma^1R_{i,j}$$. Let $$y(X) \sim N_{n}\left(\mathbf{1}_{\mathbf{n}} \mu, \Sigma\right)$$, where $$N_{n}\left(\mathbf{1}_{\mathbf{n}} \mu, \Sigma\right)$$ is a multivariate normal distribution with $$\Sigma=\sigma^{2} R$$ defined through a correlation matrix $$R$$ with elements $$R_{ij}$$ and $$\mathbf{1}_{\mathbf{n}}$$, a vector of length $$n$$ with all ones s $$n\times 1$$.

 ![GP](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/gaussian_process.png?raw=truee)
 *$$\textit{Note.}$$ Figure 7: Examplary Gaussian Process*


Figure 7 shows an exemplary GP which was generated by predicting 100 possible candidates between 1 to 10 for the hyperparameter value $$x$$ and evaluated them using an arbitrary evaluation function $$y$$, defined by equation \eqref{eq:gp_y}.
$$
\begin{equation}
y = \dfrac{a + (\sum_{i=1}^k b_i)^2 - 0.69 \times a ^3 + log(x) }{2 \times a + log(x)}
\label{eq:gp_y}
\end{equation}
$$
with $$a = \mathcal{U}(1, 10)$$ being a random variable drawn from the uniform distribution between 1 and 10  and $$b = \mathcal{N}(0,0.5^2)$$ beeing a random variable drawn from the normal distribution with mean $$0$$ and variance of $$0.5^2$$. We observe that the confidence bound increases in size, the greater the distance between two candidates who have already been evaluated. Considering that our goal would be the hyperparameter value that minimizes the evaluation function value the GP would continue sampling values around $$x = 4$$ and probably between $$x=9$$ and $$x = 10$$ since we would expect here the largest reduction in terms of the evaluation function given the estimated confidence bound.

## Metrics
Due to the different handling of factor variables by the tree-based ensemble methods the feature space can be different between methods. This is true at least when we compare a model fitted with XGBoost, which expands the feature space due to one-hot encoding, and CatBoost, which uses orderd target encoding and does not expand the feature space. In this comparison CatBoost ist always favored over XGBoost due to the factor variable handling. We have decided to use the $$RMSE$$ metric as the loss function for all gradient boosting methods and  to compare all models and choose the best model based on the $$RMSE$$ metric as well. The $$R^2$$ metric of the best model, for the prediction of the average power measurement and the UCI weekly points, will be analyzed to give a more intuitive explanation of the prediction quality of the best model.

## Tree SHAP values

A quite recent approach to interpret ensemble models was suggested by Lundberg and Lee [(2017)](https://arxiv.org/abs/1705.07874), arguing that SHAP values can be used to shed light on the inside of black-box models. The latter name is often used due to the non-parametric approach, in which we do not estimate a direct influence of a variable on the model, which leads to model results that can be used for prediction but without extensive knowledge on how the variables of the model influenced the model and the predictions. In this section, we will follow Lundberg and Lee [(2017)](https://arxiv.org/abs/1705.07874) to present a short formal explanation on how their approach works. SHAP values are based on Shapley values, first proposed by Shapley [(1952)](https://doi.org/10.7249/P0295),  who used a method from the coalition game theory to predict the payout for each player who participates. Consider each feature of a dataset as a "player", the "game" as the prediction task of the model and the "payout" as the contribution of the feature to the prediction value. SHAP values predict feature importance in the same greedy fashion as ensemble methods do. The latter means that SHAP values are calculated by refitting the model for possible feature subsets $$S \subseteq C$$, where $$C$$ is the set of all features, in an additive manner. This additive feature attribution method contains an explanation model $$g$$ that is a linear function of binary variables such that

$$
\begin{equation}
g\left(z^{\prime}\right)=\phi_{0}+\sum_{j=1}^{M} \phi_{j} z_{j}^{\prime} \\
\label{eq:shap_val}
\end{equation}
$$

where $$z^{\prime} \in \{0,1\}^M$$ is the coalition vector, $$M$$ defines the maximum coalition size and $$\phi_j \in \mathbb{R}$$ are the Shapley values.

For tree-based ensemble models, Lundberg [(2018)](https://arxiv.org/abs/1802.03888) came up with a variant of SHAP, namely TreeSHAP. The latter method uses the conditional expectation $$E_{X_{S} \mid X_{C}}\left(\hat{f}(x) \mid x_{S}\right)$$ as the value function to calculate the Shapley values. For the expected prediction of a single tree, consider an observation $$x$$ and a feature subset $$S$$ from all the set, of all features $$C$$. If $$S=C$$, the expected prediction would be equal to the prediction from the node in which the observation $$x$$ falls. If $$S=\emptyset$$ we would use the weighted mean of all predictions of all terminal nodes. If $$S \subseteq C$$, we only consider those predictions of terminal nodes that do not come from a path that contains a splitting rule dependent on feature $$\overline{S}$$, a feature that is not part of the feature subset $$S$$.

![Models overview](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/Model_presentation.png?raw=true)
*  $$\textit{Note.}$$Table 3: Overview of all models*

Table 3 gives an overview which model is used in combination with the three target variables and shows how the strava dataset of each model differ, in terms of observation size.  All models use 75\% of the data as training and 25\% for testing. Train and test sampling was random when we predicted the average power.

Since the UCI weekly points variable is updated for each athlete every week, we would probably leak some information about the values in the beginning of the distribution of the UCI weekly points variable if we were to sample training and test observations randomly as well. Therefore we decided to use the first 75\% of the observations for training and the last 25\% for testing.  For the prediction of the average power variables we will use two out-of-sample testing strategies for all models. The 5-fold cross-validation uses $$\dfrac{4}{5}$$ for training and $$\dfrac{1}{5}$$ for validation from the training sample of the strava dataset. The best model will then be used to predict on the test samples from the strava dataset. We call the latter procedure out-of-sample (OOS) testing. For the UCI weekly points variable 5-fold CV would also lead to possible leakage of information, since those folds with observations from the end would contain information about observations of the beginning of the distribution, so that we will only use the OOS testing strategy to evaluate the different methods. Furthermore, if we use the train dataset and test dataset in the upcoming results, we refer to the train and test samples from the strava dataset in described in this section.

## Results
![Models overview](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/5-fold-CV-RMSE_avg_power.png?raw=true)
*  $$\textit{Note.}$$Table 4: In-Sampele results for the average power prediction *

 The lowest RMSE value of 0.148  was achieved by XGBoost with model 3 and tuned hyperparameter settings. The reduction of the RMSE value for XGBoost with tuned hyperparameter settings was up to halve compared to the results from the default HP section. The improvement of the RMSE value of the Random Forest method with tuned hyperparameter settings compared to their RMSE results with default settings was low, with a maximum reduction of the RMSE of 0.047. In both hyperparameter sections of the table 4  we observed that the model 3 leads to the lowest error metric. Comparing the results model-wise, we discovered that, independent from the hyperparameter aspect, the models 3 \& 4, which predicted the $$\textit{avg\_power}$$ variable, achieve in total better results than model 1 \& 2 which predicted $$\textit{avg\_power\_combined}$$, a variable combined from original measurements by the bicycle computer of the athlete, and some estimations from the strava website. Moreover, model 3 seems to beats model 4 with most methods.


![OOS results power ](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/OOS_results_model_3_pow.png?raw=true)
*  $$\textit{Note.}$$ Figure 8: OOS results for the average power prediction *

For model 3 the OOS results were highlighted in figure 8. When we discussed the IS results of table 4, we argued that model 3 achieved the lowest RMSE score of all models with the XGBoost method and tuned hyperparameter settings. Related results were observed for the OOS calculation of the RMSE metric. Catboost and LGBM failed to achieve comparably low results with default hyperparameter settings while the Random Forest and XGBoost method achieve equally low RMSE values of of 0.327 and 0.336. When we used tuned hyperparameter settings to train and test model 3, we observed an error reduction almost up to 8 times for Catboost compared to the OOS results with default hyperparameter settings. With an RMSE of  0.15, the XGBoost method achieved the lowest RMSE value, as in the IS scenario.

![VIP analysis ](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/model_impact_pow.png?raw=true)
*  $$\textit{Note.}$$ Figure 9: Feature importance analysis using Tree SHAP values *

Figure \ref{fig:summary_avg_p_bar} shows that the feature $$work\_total$$ on (absolute) average added around 40 watts to the mean prediction value of the target variable $$avg\_power$$,
while $$mov\_time\_sec$$ is predicted to have the second largest impact on the prediction. If the latter feature is part of the model the average increase to the average prediction values around 20 watts. On the other hand variables with almost any contribution to the average prediction are $$UCI\_points\_weekly$$ and $$max\_cadence$$, $$type$$ and $$season$$.   It seems that the weekly tournament points, the maximal cadence value  of the training ride and the cycling type of an athlete are not helpful to predict the average power of a training ride of a professional athlete measured through a bicycle computer. The categorical variable $$season$$ seems to have no impact at all on the model. We can therefore conclude that we do not observe any seasonal influence on the average power of a training ride. All effects describe the behavior of the model and are not necessarily causal in the real world. Furthermore we do not claim that it would not matter if a road was covered with ice, nor do we claim that snow in the winter month would have no impact on the average power of a training ride compared to a training on the same road in a summer month. A reasonable explanation of these results could be, that those professional riders, which faces seasonal influence on their training routes, might train during this time of the year in the southern hemisphere or use a static bicycle trainer at home.

![SHAP summary plot ](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/model_impact_pow.png?raw=true)
*  $$\textit{Note.}$$ Figure 10: This figure shows the summary of each variable with respect to each observation of model 3 fitted with the XGBoost algorithm. All values were calculated from the test
dataset.*

Figure 11 shows that the feature $$work\_total$$ on (absolute) average added around 40 watts to the mean prediction value of the target variable $$avg\_power$$,
while $$mov\_time\_sec$$ is predicted to have the second largest impact on the prediction. If the latter feature is part of the model the average increase to the average prediction values around 20 watts. On the other hand variables with almost any contribution to the average prediction are $$UCI\_points\_weekly$$ and $$max\_cadence$$, $$type$$ and $$season$$.   It seems that the weekly tournament points, the maximal cadence value  of the training ride and the cycling type of an athlete are not helpful to predict the average power of a training ride of a professional athlete measured through a bicycle computer. The categorical variable $$season$$ seems to have no impact at all on the model. We can therefore conclude that we do not observe any seasonal influence on the average power of a training ride. All effects describe the behavior of the model and are not necessarily causal in the real world. Furthermore we do not claim that it would not matter if a road was covered with ice, nor do we claim that snow in the winter month would have no impact on the average power of a training ride compared to a training on the same road in a summer month. A reasonable explanation of these results could be, that those professional riders, which faces seasonal influence on their training routes, might train during this time of the year in the southern hemisphere or use a static bicycle trainer at home.

We will skip the results for the UCI weekly score since no method achieved a result which could be used for a usefull prediction.

The optimal hyperparamter values of model 3 fitted with the XGBoost algorithm were $$mtry = 20$$, $$Minimal node size = 40$$, $$Tree Depth = 14$$, $$learning rate = 0.0729534$$ and a $$loss reduction = 4.7306132\times10^{−4}$$.

![OOS RMSE results power w/o work load ](https://github.com/Npaffen/ds-blog/blob/main/docs/assets/OOS_power_no_workload.png?raw=true)
*$$\textit{Note.}$$Table 5*

There is a high correlation between $$avg\_power$$ and $$work\_load$$ which is likely due to the same information, power expressed in watts, was used to  calculate both variables by the bicycle computer. This might  have lead to some form of target leakage in the sense that some information about the average power is already contained in the $$work\_load$$. We considered this and used the model with the highest prediction power in terms of the lowest OOS RMSE value to predict $$avg\_power$$ again without $$work\_load$$ in the strava dataset.

 We observe that either Catboost or XGBoost are the best performing methods for all models. . With model 3 now achieving the lowest RMSE of 0.297 which equals a $$R^2$$ metric value of 0.908. So that even without the $$work\_load$$ variable we can explain more than 90\% of the variance of the predicted average power.
