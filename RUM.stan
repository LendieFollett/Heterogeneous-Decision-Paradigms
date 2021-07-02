
data {
  int N; // number of rows
  int I; // number of individuals
  int Kx; // number of columns in X
  int T; // number of individual-choice set combination
  vector<lower = 0, upper = 1>[N] y; // choice indicator
  
  real phi_prior_mean; // sensitivity analysis
  real phi_prior_sd; //
  
  matrix[N,Kx] X; // attribute matrix
  int person[N]; // index for individual
  
  int start[T]; // the starting observation for each choice set
  int end[T]; // the ending observation for each choice set
  vector[N] action; // action indicator
  vector[N] max_price; // max price in choice set
}

parameters {
  vector[5] RUMbeta;
  real<upper = 0> RUMbeta_price;
  real RUMdelta;
}

transformed parameters {
  vector[N] linpred; 
  vector[N] log_prob;
  vector[T] log_lik;
  for(idx in 1:N){
  linpred[idx]  = action[idx]*RUMdelta +
                        (X[idx,1]*RUMbeta[1])+ 
                        (X[idx,2]*RUMbeta[2])+ 
                        (X[idx,3]*RUMbeta[3])+ 
                        (X[idx,4]*RUMbeta[4])+ 
                        (X[idx,5]*RUMbeta[5])+
                        (X[idx,6])*RUMbeta_price; 
 
  }
  
  
  for (i in 1:T){
    log_prob[start[i]:end[i]] = log_softmax(linpred[start[i]:end[i]]);
    log_lik[i] = dot_product(log_prob[start[i]:end[i]], y[start[i]:end[i]]);
  }
}

model {  
    RUMdelta ~  normal(0,1);
    RUMbeta ~  normal(0, 1);
    RUMbeta_price ~ normal(0,1);
    
  // log probabilities of each choice in the dataset
  for(i in 1:T) {
    target +=log_lik[i];
  }

}


