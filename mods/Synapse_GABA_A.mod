TITLE GABA_A receptor

NEURON {
    POINT_PROCESS Synapse_GABA_A
    RANGE tau_r, tau_d, gmax, e, i, g
    NONSPECIFIC_CURRENT i
    RANGE weight
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    tau_r = 0.5 (ms) <1e-9,1e9>  : rise time
    tau_d = 8.0 (ms) <1e-9,1e9>  : decay time
    gmax = 0.0004 (uS) <0,1e9>   : max conductance
    e = -70 (mV)                 : reversal potential
    weight = 1                   : synaptic weight (unitless)
}

ASSIGNED {
    v (mV)           : membrane potential
    i (nA)           : current
    g (uS)           : conductance
    factor           : normalization factor
    tp (ms)          : time to peak
}

STATE {
    A                : activation
    B                : decay
}

INITIAL {
    A = 0
    B = 0
    
    : Calculate time to peak
    tp = (tau_r*tau_d)/(tau_d-tau_r)*log(tau_d/tau_r)
    
    : Calculate normalization factor
    factor = -exp(-tp/tau_r)+exp(-tp/tau_d)
    factor = 1/factor
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    
    : Calculate conductance
    g = gmax * weight * (B - A)
    
    : Calculate current
    i = g * (v - e)
}

DERIVATIVE state {
    A' = -A/tau_r
    B' = -B/tau_d
}

NET_RECEIVE(weight_input (unitless)) {
    LOCAL weight_scaled
    weight_scaled = weight * weight_input
    
    A = A + weight_scaled * factor
    B = B + weight_scaled * factor
}