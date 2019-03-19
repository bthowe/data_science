library(Matching)

setwd('/Users/travis.howe/Projects/github/data_science/causal_inference/I_and_R_treatment effect evaluation/matching_estimators/')


df <- read.csv(file='estimation_approaches.csv')
df[order(-df$propensity_score),]  # sort by propensity_score descending

covars <- c('one', 'two', 'three')
rr <- Match(Y=df$y, Tr=df$treatment, X=df[covars], M=1, replace=FALSE, Weight=2);

print(rr$index.treated)
print(rr$index.control)
print(rr$mdata)



# duplicate line
# command + option + up/down
# command + shift + d


# comment line
# command + shift + c


# delete line
# command + d