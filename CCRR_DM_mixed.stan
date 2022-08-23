
data {
  int N; // number of rows
  int I; // number of individuals
  int Kx; // number of columns in X
  int Kz; // number of columns in Z
  int T; // number of individual-choice set combination
  vector<lower = 0, upper = 1>[N] y; // choice indicator
  
  real phi_prior_mean; // sensitivity analysis
  real phi_prior_sd; //
  
  matrix[N,Kx] X; // attribute matrix
  matrix[I, Kz] Z; // sociodemographic matrix
  int person[N]; // index for individual
  
  int start[T]; // the starting observation for each choice set
  int end[T]; // the ending observation for each choice set
  vector[N] action; // action indicator

  matrix[N,2] trX; // attribute matrix
  matrix[N,2] gfX; // attribute matrix
  matrix[N,2] bdX; // attribute matrix
  matrix[N,2] plX; // attribute matrix
  matrix[N,2] wqX; // attribute matrix
  matrix[N,2] priceX; // attribute matrix  
  vector[N] max_price; // max price in choice set
}

parameters {
  // RR Parameters
  matrix[I,5] RRbeta;
  vector[I] RRbeta_price;
  vector<lower = 0>[5] RR_mu_beta; // mean of random parameters
  real<upper = 0> RR_mu_beta_price;// mean of random parameters
  real<lower = 0>  RRmu;
  real RRdelta;
  vector<lower = 0>[5]RR_sigma_beta;
  real<lower = 0>RR_sigma_beta_price;
  
   // CC Parameters
  matrix[I,5] CCbeta;
  vector[I] CCbeta_price;
  vector<lower = 0>[5] CC_mu_beta;
  real<lower = 0> CC_mu_beta_price;
  real CCdelta;
  real<lower = 0> phi_price;
  real<lower = 0> phi_tr;
  real<lower = 0> phi_gf;
  real<lower = 0> phi_bd;
  real<lower = 0> phi_pl;
  real<lower = 0> phi_wq;
  vector<lower = 0>[5]CC_sigma_beta;
  real<lower = 0>CC_sigma_beta_price;
      
  // Class membership specification parameters
  vector[Kz] gamma;
  
  

}

transformed parameters {
  vector[N] CCMlog_prob;
  vector[N] RRlog_prob;
  vector[I] rho;
  vector[N] CCMlinpred; 
  vector[N] RRlinpred; 
  vector[T] log_lik;
    
  for(idx in 1:N){
  CCMlinpred[idx]  = action[idx]*CCdelta +
                        CCbeta[person[idx],1]*(X[idx,1]^phi_tr)+ 
                        CCbeta[person[idx],2]*(X[idx,2]^phi_gf)+ 
                        CCbeta[person[idx],3]*(X[idx,3]^phi_bd)+ 
                        CCbeta[person[idx],4]*(X[idx,4]^phi_pl)+ 
                        CCbeta[person[idx],5]*(X[idx,5]^phi_wq)+
                        CCbeta_price[person[idx]]*((max_price[idx]-X[idx,6] )^phi_price); 
        // print(phi_price);                    
  RRlinpred[idx]  =  action[idx]*RRdelta +
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
  
  for (i in 1:I){
    rho[i] = inv_logit(Z[i,]*gamma);
  }
 
 
   for(i in 1:T) {
    CCMlog_prob[start[i]:end[i]] = log_softmax(CCMlinpred[start[i]:end[i]]);
    RRlog_prob[start[i]:end[i]] =  log_softmax(-RRlinpred[start[i]:end[i]]);
    log_lik[i]= log_mix(rho[person[start[i]]],
              dot_product(CCMlog_prob[start[i]:end[i]],y[start[i]:end[i]]),
              dot_product(RRlog_prob[start[i]:end[i]],y[start[i]:end[i]]));
   }
    
//        RUM_ind_probs[i] = exp(dot_product(RUMlog_prob[start[i]:end[i]],y[start[i]:end[i]]))*rho[person[start[i]]] / 
//    (exp(dot_product(RUMlog_prob[start[i]:end[i]],y[start[i]:end[i]]))*rho[person[start[i]]] + exp(dot_product(RRlog_prob[start[i]:end[i]],y[start[i]:end[i]]))*(1-rho[person[start[i]]]));
   

 
  
}

model {  
  
RR_mu_beta ~  normal(0,.25);
CC_mu_beta ~  normal(0,.25);
RR_mu_beta_price ~  normal(0,.25);
CC_mu_beta_price ~  normal(0,.25);
CC_sigma_beta ~ normal(0, .1);
RR_sigma_beta ~ normal(0, .1);
CC_sigma_beta_price~ normal(0, .1);
RR_sigma_beta_price~ normal(0, .1);

for (k in 1:5){
    CCbeta[,k] ~  normal(CC_mu_beta[k], CC_sigma_beta[k]);
    RRbeta[,k] ~  normal(RR_mu_beta[k], RR_sigma_beta[k]);
}
    CCbeta_price ~ normal(CC_mu_beta_price, CC_sigma_beta_price);
    RRbeta_price ~ normal(RR_mu_beta_price, RR_sigma_beta_price);
    
    CCdelta ~  normal(0,.25);
    RRdelta ~  normal(0,.25);
    gamma ~ normal(0,.25);
    RRmu ~ normal(0,1);
    
    phi_tr ~ normal(phi_prior_mean, phi_prior_sd);
    phi_gf ~ normal(phi_prior_mean, phi_prior_sd);
    phi_bd ~ normal(phi_prior_mean, phi_prior_sd);
    phi_pl ~ normal(phi_prior_mean, phi_prior_sd);
    phi_wq ~ normal(phi_prior_mean, phi_prior_sd);
    phi_price ~ normal(phi_prior_mean, phi_prior_sd);
    
  // log probabilities of each choice in the dataset
  for(i in 1:T) {
    target += log_lik[i];
  }
  

}
