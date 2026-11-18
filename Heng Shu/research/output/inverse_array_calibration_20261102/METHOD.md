> Historical workflow documentation. Generated results and private input manifests referenced below are not bundled. See the top-level README for current package status.

# Shared-phase noise-subspace fitting (project-specific adaptation)

This is a diagnostic inverse model, not certified hardware calibration. A constant phase per physical antenna represents the combined antenna/feedline/switch/RF response. Antenna 1 fixes the gauge. The same phase vector is used in both calibration positions and all six states; each record's global per-chain pi choice is removed first.

## Observation and ambiguity

H_{r,s,m,k,p} = exp(j alpha_{r,s,p} - j 2 pi Deltaf_k delta_{r,s,p})
 * g_{s,m}(f_k) * sum_l beta_{r,s,l,p} exp(-j 2 pi f_k L_{r,l,m}/c) + n.

Common packet errors are distinct from fixed relative antenna phase, phase slope, and path-specific propagation. Multipath remains after known-TX propagation removal. The bounded fit tests g_m=exp(j phi_m); it does not assume that constant-phase adequacy or cross-record hardware constancy has been proven. A pi branch correction alone does not prove a common hardware session.

All shared-antenna constraints are within-state relative measurements. No inter-state observations are interpolated or assembled as simultaneous array samples. Twelve reference-antenna edges connect twelve physical antennas, with one independent graph cycle.

## Dominant-path circular fit

For an explicitly assumed known path,
q_{r,s,j,k,p}=unit(H_j H_1*) / pi_branch_{r,j}
v_{r,s,j}=mean_{p,k}[q * exp(+j 2 pi f_k (L_j-L_1)/c)].
Fit phi_1=0 by minimizing sum |exp(j(phi_j-phi_1state))-v|^2.
This uses the magnitude of the circular mean implicitly: diffuse means exert less phase preference.
Compare the two calibration positions' residual phase constraints before claiming fixed hardware response.
The dominant-human assumption is a hypothesis, not a path label.

## Subspace construction

Use the last704 cached selected packets per record/state, after existing conditional common-error processing and global chain pi normalization.
R = mean_{16 frequency windows, packets} z z^H.
Each z is three antennas times fifteen contiguous reported HT40 tones (45 dimensions).
The strongest q=3 eigenvectors form E_s. Keep all states separate.
Frequency smoothing and q follow the fixed baseline. Neither is tuned on validation geometry.

For each hypothesized path h and each state:
L_m^TX=||TX-r_m||,
L_m^human=||TX-p||+||p-r_m||.
b_h = repeat_m(exp(-j 2 pi f_c (L_m-L_ref)/c),15)
      * exp(-j 2 pi Deltaf (L_ref/c + delta_rs)) / sqrt(45).
z_h(phi,delta)=diag(repeat_m(exp(j phi_m),15)) b_h.

J = mean_{calibration record, state, hypothesized path}
    [1 - || E_s^H z_h ||^2].

There are11 shared phase unknowns and12 common reference delays. In TX+human fits, both paths share the same delta_rs. The calibration human positions are used only for records0 and12; the mirror placebo reverses their RX-local y coordinate. Each delta is bounded to one800ns period. It is a nuisance reference; it is not a recovered physical path delay.

This steering retains exact spherical carrier-frequency relative path phases but applies the baseline narrowband spatial approximation during frequency smoothing. It is not an exact full-band spherical-wave likelihood.

## Bounded optimizer and numerical checks

Eight deterministic starts per model: zero, dominant-TX solution, dominant-human solution, five seeded random phase vectors. Initial delay selected over a fixed8ns grid per calibration record/state, from calibration loss alone.
L-BFGS-B, maximum300 iterations, analytic gradients; no holdout-based restart/selection.
Original ns scaling and the documented delta=20*u ns reparameterization are both preserved.
All32 scaled fits converged locally; this is not a global optimality certificate.
Gradient checked by central finite difference; connected graph solver checked on synthetic exact known phases.
Near-optimal solution spread uses best+max(1% of best,1e-4) only as a descriptive grouping.

## Application and prediction

For a state with physical antennas m, D=diag(repeat(exp(-j phi_m),15)).
R_cal=D R_pi D^H. Covariance eigenvalues are preserved by this unitary transformation.
Compute fixed-q MUSIC with the physical state steering vector on the baseline angle/delay grid.
Marginalize each state's unknown delay by maximum and fuse G1-G4 angular spectra with the previously frozen data-quality weights. C1/C2 are calibration constraints and diagnostic spectra, not extra duplicated fusion weight.
Do not coherently concatenate states.

Predictions and file hashes are saved before the evaluation module reads withheld coordinates.
The strongest fusion peak is unassigned. Without independent human-path attribution, comparison to geometric truth is a diagnostic deviation rather than human AoA MAE.
The mirror control is compared with equal two-path constraint count. Losses for one-path versus two-path hypotheses are not a formal model selection test.
No delay-to-position inversion, no physical error metric, and no new full-array localization claim.

## Limitations of this bounded comparison

A small number of cached records, previously inspected retrospective data, unverified pi/constant-phase hardware assumptions, one fixed q, approximate frequency smoothing, local optimization, and an unassigned strongest-peak output limit the conclusions.
Failure here does not establish that all estimators or all remaining raw data are incapable of localization.
Independent fixed-response calibration and presence/absence controls remain missing.
