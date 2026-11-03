Compact regression model definitions retained from historical project tests.

les.mtp       : 29-moment short-range seed for LES/SOG and trainer tests
plain.mtp     : 29-moment fitted plain MTP
nonlinear.mtp : 92-moment MTP, including nonlinear moment products
hybrid.mtp    : short-range basis definition used by hybrid tests

The nonlinear model exercises growth of moment storage after loading a smaller
model, without the historical 82 MB fixture. These are numerical regression
inputs, not validated production potentials. See ../README.md for data scope
and the LES seed update.
