
data {
  int N; // number of rows
  int I; // number of individuals
  int Kx; // number of columns in X
  int T; // number of individual-choice set combination
  vector<lower = 0, upper = 1>[N] y; // choice indicator
  matrix[N,2] trX; // attribute matrix
    matrix[N,2] gfX; // attribute matrix
      matrix[N,2] bdX; // attribute matrix
        matrix[N,2] plX; // attribute matrix
          matrix[N,2] wqX; // attribute matrix
          matrix[N,2] priceX; // attribute matrix          
  int person[N]; // index for individual
  
  int start[T]; // the starting observation for each choice set
  int end[T]; // the ending observation for each choice set
  vector[N] action; // action indicator

}

parameters {
  matrix[I,5] RRbeta; // I random parameters
  real<lower = 0>  RRmu;
  vector[I] RRbeta_price;
  real delta;
  
  vector<lower = 0>[5] RR_mu_beta; // mean of random parameters
  real<upper = 0> RR_mu_beta_price;// mean of random parameters
  vector<lower = 0>[5]RR_sigma_beta;
  real<lower = 0>RR_sigma_beta_price;
}

transformed parameters {
  vector[N] linpred; 
  vector[N] log_prob;
  vector[T] log_lik;
  for(idx in 1:N){
  linpred[idx]  =  action[idx]*delta +
            RRmu*log(1 + exp(trX[idx,1]*RRbeta[person[idx],1]/RRmu)) + 
            RRmu*log(1 + exp(trX[idx,2]*RRbeta[person[idx],1]/RRmu))+                   
            RRmu*log(1 + exp(gfX[idx,1]*RRbeta[person[idx],2]/RRmu)) + 
            RRmu*log(1 + exp(gfX[idx,2]*RRbeta[person[idx],2]/RRmu))+ 
            RRmu*log(1 + exp(bdX[idx,1]*RRbeta[person[idx],3]/RRmu)) + 
            RRmu*log(1 + exp(bdX[idx,2]*RRbeta[person[idx],3]/RRmu))+ 
            RRmu*log(1 + exp(plX[idx,1]*RRbeta[person[idx],4]/RRmu)) + 
            RRmu*log(1 + exp(plX[idx,2]*RRbeta[person[idx],4]/RRmu))+ 
            RRmu*log(1 + exp(wqX[idx,1]*RRbeta[person[idx],5]/RRmu)) + 
            RRmu*log(1 + exp(wqX[idx,2]*RRbeta[person[idx],5]/RRmu))+
            RRmu*log(1 + exp(priceX[idx,1]*RRbeta_price[person[idx]]/RRmu)) + 
            RRmu*log(1 + exp(priceX[idx,2]*RRbeta_price[person[idx]]/RRmu));
  }
    for (i in 1:T){
    log_prob[start[i]:end[i]] = log_softmax(-linpred[start[i]:end[i]]);
    log_lik[i] = dot_product(log_prob[start[i]:end[i]], y[start[i]:end[i]]);
  }
  
}

model {  
    delta ~ normal(0, .25);
for (k in 1:5){
    RRbeta[,k] ~  normal(RR_mu_beta[k], RR_sigma_beta[k]);
}
    RRbeta_price ~ normal(RR_mu_beta_price, RR_sigma_beta_price);

    RRmu ~ normal(0,1);
    
    RR_mu_beta ~  normal(0,.25);

RR_mu_beta_price ~  normal(0,.25);
RR_sigma_beta ~ normal(0, .1);
RR_sigma_beta_price~ normal(0, .1);

  for(i in 1:T) {
    target +=log_lik[i];
  }

}


