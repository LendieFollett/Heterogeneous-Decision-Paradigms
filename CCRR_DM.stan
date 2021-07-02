data {
  int N;  // number of rows
  int I;  // number of individuals
  int Kx; // number of columns in X
  int Kz; // number of columns in Z
  int T;  // number of individual-choice set combination
  vector<lower = 0, upper = 1>[N] y; // choice indicator
  
  real phi_prior_mean; // phi prior parameters
  real phi_prior_sd;  
  
  matrix[N,Kx] X;  // attribute matrix
  matrix[I, Kz] Z; // sociodemographic matrix
  int person[N];   // index for individual
  
  int start[T];     // the starting observation for each choice set
  int end[T];       // the ending observation for each choice set
  vector[N] action; // action indicator

  matrix[N,2] trX;    // RR alternatives matrix
  matrix[N,2] gfX;    // RR alternatives matrix
  matrix[N,2] bdX;    // RR alternatives matrix
  matrix[N,2] plX;    // RR alternatives matrix
  matrix[N,2] wqX;    // RR alternatives matrix
  matrix[N,2] priceX; // RR alternatives matrix  
  
  vector[N] max_price; // max price in choice set
}

parameters {
  vector<lower = 0>[5] RRbeta;
  vector<lower = 0>[5] CCbeta;
  real<upper = 0> RRbeta_price;
  real<lower = 0> CCbeta_price;
  real RRdelta;
  real CCdelta;
  real<lower = 0> phi_price;
  real<lower = 0> phi_tr;
  real<lower = 0> phi_gf;
  real<lower = 0> phi_bd;
  real<lower = 0> phi_pl;
  real<lower = 0> phi_wq;
  vector[Kz] gamma;
  //vector[I] alpha;
 // real<lower = 0> sigma_alpha;
  //real mu_alpha;
}

transformed parameters {
  vector[N] CCMlog_prob;//CCM choice probabilities
  vector[N] RRlog_prob; //RR choice probabilities
  vector[I] rho;        //Membership probability
  vector[N] CCMlinpred; //CCM utility function
  vector[N] RRlinpred;  //RRM regret function
  vector[T] log_lik;    //For LOO calculation
    
  for(idx in 1:N){
  CCMlinpred[idx]  = action[idx]*CCdelta +
                        CCbeta[1]*(X[idx,1]^phi_tr)+ 
                        CCbeta[2]*(X[idx,2]^phi_gf)+ 
                        CCbeta[3]*(X[idx,3]^phi_bd)+ 
                        CCbeta[4]*(X[idx,4]^phi_pl)+ 
                        CCbeta[5]*(X[idx,5]^phi_wq)+
                        CCbeta_price*((max_price[idx]-X[idx,6] )^phi_price); 
                  
  RRlinpred[idx]  =  action[idx]*RRdelta +
            log(1 + exp(trX[idx,1]*RRbeta[1])) + 
            log(1 + exp(trX[idx,2]*RRbeta[1]))+                   
            log(1 + exp(gfX[idx,1]*RRbeta[2])) + 
            log(1 + exp(gfX[idx,2]*RRbeta[2]))+ 
            log(1 + exp(bdX[idx,1]*RRbeta[3])) + 
            log(1 + exp(bdX[idx,2]*RRbeta[3]))+ 
            log(1 + exp(plX[idx,1]*RRbeta[4])) + 
            log(1 + exp(plX[idx,2]*RRbeta[4]))+ 
            log(1 + exp(wqX[idx,1]*RRbeta[5])) + 
            log(1 + exp(wqX[idx,2]*RRbeta[5]))+
            log(1 + exp(priceX[idx,1]*RRbeta_price)) + 
            log(1 + exp(priceX[idx,2]*RRbeta_price)) ;
  }                      
  
  for (i in 1:I){
    rho[i] = inv_logit( Z[i,]*gamma );//++ alpha[i] 
  }
 
  for(i in 1:T) {
    CCMlog_prob[start[i]:end[i]] = log_softmax(CCMlinpred[start[i]:end[i]]);
    RRlog_prob[start[i]:end[i]] =  log_softmax(-RRlinpred[start[i]:end[i]]);
    log_lik[i]= log_mix(rho[person[start[i]]],
              dot_product(CCMlog_prob[start[i]:end[i]],y[start[i]:end[i]]),
              dot_product(RRlog_prob[start[i]:end[i]],y[start[i]:end[i]]));
   }
}

model {  
    CCdelta ~  normal(0,1);
    RRdelta ~  normal(0,1);
    CCbeta ~  normal(0, 1);
    RRbeta ~  normal(0, 1);
    CCbeta_price ~ normal(0, 1);
    RRbeta_price ~ normal(0, 1);
    //gamma ~ normal(0,1);
    //alpha ~ normal(0,sigma_alpha);
    //sigma_alpha ~ normal(0, .5);
    gamma ~ normal(0, 0.5);
    
    phi_tr ~ normal(phi_prior_mean, phi_prior_sd);
    phi_gf ~ normal(phi_prior_mean, phi_prior_sd);
    phi_bd ~ normal(phi_prior_mean, phi_prior_sd);
    phi_pl ~ normal(phi_prior_mean, phi_prior_sd);
    phi_wq ~ normal(phi_prior_mean, phi_prior_sd);
    phi_price ~ normal(phi_prior_mean, phi_prior_sd);
    
  for(i in 1:T) {
    target += log_lik[i];
  }
  
}
