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
    vector[N] max_price; // max price in choice task

}

parameters {
  vector<lower = 0>[5] CCbeta;
  real<lower = 0> CCbeta_price;
  real CCdelta;
  real<lower = 0> phi_price;
  real<lower = 0> phi_tr;
  real<lower = 0> phi_gf;
  real<lower = 0> phi_bd;
  real<lower = 0> phi_pl;
  real<lower = 0> phi_wq;
}

transformed parameters {
  vector[N] linpred; 
  vector[N] log_prob;
  vector[T] log_lik;
  for(idx in 1:N){
    //Chorus Bierlaire 2012 form
    // x - least preferred level of x in given choice task
  linpred[idx]  = action[idx]*CCdelta +
                        CCbeta[1]*(X[idx,1]^phi_tr)+ 
                        CCbeta[2]*(X[idx,2]^phi_gf)+ 
                        CCbeta[3]*(X[idx,3]^phi_bd)+ 
                        CCbeta[4]*(X[idx,4]^phi_pl)+ 
                        CCbeta[5]*(X[idx,5]^phi_wq)+
                        CCbeta_price*((max_price[idx]-X[idx,6])^phi_price); 
 
  }
  for (i in 1:T){
    log_prob[start[i]:end[i]] = log_softmax(linpred[start[i]:end[i]]);
    log_lik[i] = dot_product(log_prob[start[i]:end[i]], y[start[i]:end[i]]);
  }
}

model {  

    phi_tr ~ normal(phi_prior_mean, phi_prior_sd);
    phi_gf ~ normal(phi_prior_mean, phi_prior_sd);
    phi_bd ~ normal(phi_prior_mean, phi_prior_sd);
    phi_pl ~ normal(phi_prior_mean, phi_prior_sd);
    phi_wq ~ normal(phi_prior_mean, phi_prior_sd);
    phi_price ~ normal(phi_prior_mean, phi_prior_sd);
    CCdelta ~  normal(0,1);
    CCbeta ~   normal(0,1);
    CCbeta_price ~ normal(0,1);
    
  // log probabilities of each choice in the dataset
  for(i in 1:T) {
    target +=log_lik[i];
  }
  // y*log(exp(eta)/sum(exp(eta))), sum them

}


