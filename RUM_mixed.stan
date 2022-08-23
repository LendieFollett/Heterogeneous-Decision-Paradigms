
data {
  int N; // number of rows
  int I; // number of individuals
  int Kx; // number of columns in X
  int T; // number of individual-choice set combination
  vector<lower = 0, upper = 1>[N] y; // choice indicator
  
  matrix[N,Kx] X; // attribute matrix
  int person[N]; // index for individual
  
  int start[T]; // the starting observation for each choice set
  int end[T]; // the ending observation for each choice set
  vector[N] action; // action indicator
  vector[N] max_price; // max price in choice set
}

parameters {
  matrix[I,5] RUMbeta;
  vector[I] RUMbeta_price;
  real RUMdelta;
  vector[5] RUM_mu_beta;
  real <upper = 0>RUM_mu_beta_price;
  vector<lower = 0>[5]RUM_sigma_beta;
  real<lower = 0>RUM_sigma_beta_price;
}

transformed parameters {
  vector[N] linpred; 
  vector[N] log_prob;
  vector[T] log_lik;
  for(idx in 1:N){
  linpred[idx]  = action[idx]*RUMdelta +
                        (X[idx,1])*RUMbeta[person[idx],1] + 
                        (X[idx,2])*RUMbeta[person[idx],2] + 
                        (X[idx,3])*RUMbeta[person[idx],3] + 
                        (X[idx,4])*RUMbeta[person[idx],4] + 
                        (X[idx,5])*RUMbeta[person[idx],5] +
                        (X[idx,6])*RUMbeta_price[person[idx]]; 
 
  }
  
  
  for (i in 1:T){
    log_prob[start[i]:end[i]] = log_softmax(linpred[start[i]:end[i]]);
    log_lik[i] = dot_product(log_prob[start[i]:end[i]], y[start[i]:end[i]]);
  }
}

model {  
RUMdelta ~  normal(0,.25);
RUM_mu_beta ~  normal(0,.25);
RUM_mu_beta_price ~  normal(0,.25);
RUM_sigma_beta ~ normal(0, .1);
RUM_sigma_beta_price~ normal(0, .1);

for (k in 1:5){
    RUMbeta[,k] ~  normal(RUM_mu_beta[k], RUM_sigma_beta[k]);
}
RUMbeta_price ~ normal(RUM_mu_beta_price, RUM_sigma_beta_price); 

  // log probabilities of each choice in the dataset
  for(i in 1:T) {
    target +=log_lik[i];
  }

}


