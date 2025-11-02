TITLE AMPA and NMDA receptor with magnesium block

NEURON {
    POINT_PROCESS Synapse_AMPA_NMDA
    RANGE tau_r_AMPA, tau_d_AMPA, gmax_AMPA, e_AMPA, i_AMPA, g_AMPA
    RANGE tau_r_NMDA, tau_d_NMDA, gmax_NMDA, e_NMDA, i_NMDA, g_NMDA
    RANGE mg, gamma, mggate, i_syn, g_syn
    NONSPECIFIC_CURRENT i_AMPA, i_NMDA
    RANGE weight
}

UNITS {
    (nA)    = (nanoamp)
    (mV)    = (millivolt)
    (uS)    = (microsiemens)
	(molar) = (mole/litre)
    (mM)    = (millimolar)
}

PARAMETER {
    tau_r_AMPA = 0.3 (ms) <1e-9,1e9>  : AMPA rise time
    tau_d_AMPA = 3.0 (ms) <1e-9,1e9>  : AMPA decay time
    gmax_AMPA = 0.0004 (uS) <0,1e9>   : AMPA max conductance
    e_AMPA = 0 (mV)                   : AMPA reversal potential
    
    tau_r_NMDA = 2.0 (ms) <1e-9,1e9>  : NMDA rise time
    tau_d_NMDA = 70.0 (ms) <1e-9,1e9> : NMDA decay time
    gmax_NMDA = 0.0004 (uS) <0,1e9>   : NMDA max conductance
    e_NMDA = 0 (mV)                   : NMDA reversal potential
    
    mg = 1.0 (mM)                     : external magnesium concentration
    gamma = 0.062 (/mV)               : steepness of voltage-dependent Mg block
    weight = 1                        : synaptic weight (unitless)
}

ASSIGNED {
    v (mV)                : membrane potential
    i_AMPA (nA)           : AMPA current
    i_NMDA (nA)           : NMDA current
    i_syn (nA)            : total synaptic current
    g_AMPA (uS)           : AMPA conductance
    g_NMDA (uS)           : NMDA conductance
    g_syn (uS)            : total conductance
    mggate                : mgblock variable
    factor_AMPA           : AMPA normalization factor
    factor_NMDA           : NMDA normalization factor
    tp_AMPA (ms)          : time to peak for AMPA
    tp_NMDA (ms)          : time to peak for NMDA
}

STATE {
    A_AMPA                : AMPA activation
    B_AMPA                : AMPA decay
    A_NMDA                : NMDA activation
    B_NMDA                : NMDA decay
}

INITIAL {
    A_AMPA = 0
    B_AMPA = 0
    A_NMDA = 0
    B_NMDA = 0
    
    mggate = 1 / (1 + (mg/3.57 (mM)) * exp(-gamma * v))
    
    : Calculate time to peak using the same method as in ProbAMPANMDA2
    tp_AMPA = (tau_r_AMPA*tau_d_AMPA)/(tau_d_AMPA-tau_r_AMPA)*log(tau_d_AMPA/tau_r_AMPA)
    tp_NMDA = (tau_r_NMDA*tau_d_NMDA)/(tau_d_NMDA-tau_r_NMDA)*log(tau_d_NMDA/tau_r_NMDA)
    
    : Calculate normalization factors as in ProbAMPANMDA2
    factor_AMPA = -exp(-tp_AMPA/tau_r_AMPA)+exp(-tp_AMPA/tau_d_AMPA)
    factor_AMPA = 1/factor_AMPA
    
    factor_NMDA = -exp(-tp_NMDA/tau_r_NMDA)+exp(-tp_NMDA/tau_d_NMDA)
    factor_NMDA = 1/factor_NMDA
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    
    : Update the magnesium block gate based on voltage
    mggate = 1 / (1 + (mg/3.57 (mM)) * exp(-gamma * v))
    
    : Calculate conductances
    g_AMPA = gmax_AMPA * weight * (B_AMPA - A_AMPA)
    g_NMDA = gmax_NMDA * weight * (B_NMDA - A_NMDA) * mggate
    g_syn = g_AMPA + g_NMDA
    
    : Calculate currents
    i_AMPA = g_AMPA * (v - e_AMPA)
    i_NMDA = g_NMDA * (v - e_NMDA)
    i_syn = i_AMPA + i_NMDA
}

DERIVATIVE state {
    A_AMPA' = -A_AMPA/tau_r_AMPA
    B_AMPA' = -B_AMPA/tau_d_AMPA
    A_NMDA' = -A_NMDA/tau_r_NMDA
    B_NMDA' = -B_NMDA/tau_d_NMDA
}

NET_RECEIVE(weight_input (unitless)) {
    LOCAL weight_scaled
    weight_scaled = weight * weight_input
    
    A_AMPA = A_AMPA + weight_scaled * factor_AMPA
    B_AMPA = B_AMPA + weight_scaled * factor_AMPA
    A_NMDA = A_NMDA + weight_scaled * factor_NMDA
    B_NMDA = B_NMDA + weight_scaled * factor_NMDA
}