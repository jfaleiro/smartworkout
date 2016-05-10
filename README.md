# smartworkout

**[CLICK HERE FOR THE HTML ANALYSIS](https://cdn.rawgit.com/jfaleiro/smartworkout/master/index.html)**

People regularly quantify _*how much*_ of a particular activity they do, but they rarely quantify _*how well*_ that same activity is performed. More often than not discerning the quality of a work out requires specialized supervision of a personal trainer.

Have you ever imagined a scenario in which your training equipment would play the role of your personal trainer?

This is actually what this whole analysis is all about. We predict how well people exercise based on data produced by accelarators attached to their belt, forearm, arm, and dumbell. 

The overall quality in which people exercise is given by the "classe" variable in the training set. Classe 'A' indicates an exercise perfomed correctly (all kudos to you, athlete). The other classes indicate common exercizing mistakes.

All credits for data collection and original analysis go to the [Human Activity Recognition - HAR](http://groupware.les.inf.puc-rio.br/har) laboratory, previously detailed in [this paper][1]. Credits for educational notes go to the [Johns Hopkins School of Biostatistics](http://www.jhsph.edu/departments/biostatistics/).

In a simple and quick fitting we were able to get very close to the weighted average of the [baseline accuracy][1] of **99.4%**. Despite of the numerical proximity of the results, we can see the baseline is on the upper boundary of the confidence interval of this study.

We were limited in terms of computing resources and time (this analysis was performed beginning to end in about 3 hours). If we had more time we could try ensemble methods for classifications, specifically `AdaBoost`, but that would be beyond the intent and time allocated for this exercise.

If you want to check a more elaborate analysis you can either check the [original paper][1] or refer to  a [longer version of this study][3], where we list several techniques and options for investigation over the same raw data.

[1]: http://groupware.les.inf.puc-rio.br/public/papers/2012.Ugulino.WearableComputing.HAR.Classifier.RIBBON.pdf 
[2]: http://topepo.github.io/caret/Implicit_Feature_Selection.html
[3]: http://rpubs.com/jfaleiro/smartbells
