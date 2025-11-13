

version 17
clear all
set more off
capture log close
cd "/Users/alexelshyeb/Downloads/640"
log using "hw_3.txt", text replace

set seed 12345

* parameters
local R   = 1000 //replications
local G   = 50   // clusters
local T   = 10   // time periods per cluster
local rho = 0.5  // autoregression coefficient
local a   = 0  // true alpha
local b   = 1  // true beta

* one replication
capture program drop oneSim
program define oneSim, rclass //returns scalars
    version 17
    args G T rho a b  //arguments 

    clear
    set obs `G'  // creates G clusters
    gen i = _n

    // σ_i^2 ~ U[0.5, 2], σ_i = sqrt(σ_i^2)
    gen var_i   = runiform(0.5, 2)  // cluster specific variance
    gen sigma_i = sqrt(var_i)

    expand `T'  // expand clusters to T time periods each
    bysort i: gen t = _n
    tsset i t  // creates panel

    // regressor x, error eps
    gen x   = rnormal()   //exogenous regressor
    gen eps = rnormal()   // error

    // error within cluster
    gen u = .
    bysort i (t): replace u = sigma_i*eps if t==1 //heteroskedasticity across clusters
    bysort i (t): replace u = `rho'*L.u + sigma_i*eps if t>1  //serial correlation within clusters, L.u generates lagged u variable (requires tsset)

    // model
    gen y = `a' + `b'*x + u  // y is coutcome variable

    // 1: OLS 
    regress y x
    return scalar b_ols  = _b[x]
    return scalar se_ols = _se[x]

    // 2: OLS clustered
    regress y x, vce(cluster i)
    return scalar b_c_ols  = _b[x]
    return scalar se_c_ols = _se[x]

    // 3: GLS/FGLS
    xtset i t
    xtgls y x, panels(hetero) corr(ar1)
    return scalar b_gls  = _b[x]
    return scalar se_gls = _se[x]
end

* Monte Carlo
simulate                                                     ///
    b_ols=r(b_ols) se_ols=r(se_ols)                          ///
    b_c_ols=r(b_c_ols) se_c_ols=r(se_c_ols)                  ///
    b_gls=r(b_gls)   se_gls=r(se_gls),                       ///
    reps(`R') seed(20250101) nodots:                         ///
    oneSim `G' `T' `rho' `a' `b'

* Summaries
display "==== Point estimates (mean, sd) over `R' replications ===="
quietly su b_ols
display "OLS:           mean(b) = " %6.3f r(mean) ", sd(b) = " %6.3f r(sd)
quietly su b_c_ols
display "Clustered OLS: mean(b) = " %6.3f r(mean) ", sd(b) = " %6.3f r(sd)
quietly su b_gls
display "GLS/FGLS:      mean(b) = " %6.3f r(mean) ", sd(b) = " %6.3f r(sd)

display "==== Model-based SEs (mean across `R' reps) ===="
quietly su se_ols
display "OLS SE (conventional):          mean(SE) = " %6.3f r(mean)
quietly su se_c_ols
display "Clustered OLS SE (cluster i):   mean(SE) = " %6.3f r(mean)
quietly su se_gls
display "GLS/FGLS SE (model-based):      mean(SE) = " %6.3f r(mean)

display "==== Empirical sd(b) vs mean SE ===="
quietly su b_ols
local sd_b_ols = r(sd)
quietly su se_ols
display "OLS:           sd(b) = " %6.3f `sd_b_ols' ",  mean(SE) = " %6.3f r(mean)

quietly su b_c_ols
local sd_b_c_ols = r(sd)
quietly su se_c_ols
display "Clustered OLS: sd(b) = " %6.3f `sd_b_c_ols' ",  mean(SE) = " %6.3f r(mean)

quietly su b_gls
local sd_b_gls = r(sd)
quietly su se_gls
display "GLS/FGLS:      sd(b) = " %6.3f `sd_b_gls' ",  mean(SE) = " %6.3f r(mean)

* histograms:
histogram b_ols,    name(h1, replace) title("β̂ OLS")
histogram b_c_ols,  name(h2, replace) title("β̂ Clustered OLS")
histogram b_gls,    name(h3, replace) title("β̂ GLS/FGLS")

***** save *****

* make a wd
capture mkdir "figures"

* OLS histogram
histogram b_ols, width(0.02) title("β̂ OLS") name(h1, replace)
graph export "figures/hist_ols.png", replace

* Clustered OLS histogram
histogram b_c_ols, width(0.02) title("β̂ Clustered OLS") name(h2, replace)
graph export "figures/hist_clustered_ols.png", replace

* GLS/FGLS histogram
histogram b_gls, width(0.02) title("β̂ GLS/FGLS") name(h3, replace)
graph export "figures/hist_gls.png", replace

log close
