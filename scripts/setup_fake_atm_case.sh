#!/bin/bash
# =============================================================================
# setup_fake_atm_case.sh
#
# Creates and configures a CESM2.1.5 GIAF case (POP2 ocean + CICE + DATM
# in CAMULATOR mode) using wchapman's fork which has the CAMULATOR datamode
# already wired into DATM.
#
# Usage:
#   bash setup_fake_atm_case.sh           # create + setup + build
#   bash setup_fake_atm_case.sh nobuild   # create + setup only
#
# Prerequisites:
#   1. Derecho allocation — set PROJECT below
#   2. fake_atm_server.py running on Casper GPU before CESM starts
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION — update PROJECT when you have Derecho allocation
# =============================================================================
CESM_ROOT=/glade/work/wchapman/JE_help_cnn/camulator_sandbox
CASE_DIR=/glade/work/praggarwal/cesm/CREDIT/g.e21.FAKE_ATM_GIAF_v01
RUNDIR=/glade/derecho/scratch/praggarwal/g.e21.FAKE_ATM_GIAF_v01/run
PROJECT=UCSD0044          # <-- your Derecho project code (get from PI)
MACH=derecho
COMPILER=intel
COMPSET=GIAF              # active POP2 + CICE + DATM IAF
RES=T62_g17               # T62 atm (94×192) matches our MLP grid; gx1v7 ocean

# =============================================================================
# STEP 1 — create_newcase
# =============================================================================
echo "==> Creating case: $CASE_DIR"
${CESM_ROOT}/cime/scripts/create_newcase \
    --case     ${CASE_DIR} \
    --mach     ${MACH} \
    --compiler ${COMPILER} \
    --compset  ${COMPSET} \
    --res      ${RES} \
    --project  ${PROJECT} \
    --run-unsupported

cd ${CASE_DIR}

# =============================================================================
# STEP 2 — xmlchanges
# =============================================================================
echo "==> Applying xmlchanges..."

# DATM in CAMULATOR mode (file-based handshake with fake_atm_server.py)
./xmlchange DATM_MODE=CAMULATOR

# 6-hour coupling interval
./xmlchange NCPL_BASE_PERIOD=day
./xmlchange ATM_NCPL=4
./xmlchange OCN_NCPL=4

# 1-year test run (no resubmit — we check stability first)
./xmlchange STOP_OPTION=nyears
./xmlchange STOP_N=1
./xmlchange RESUBMIT=0

# Derecho queue and walltime (~38 min for 1 year at 256 cores)
./xmlchange JOB_QUEUE=main
./xmlchange JOB_WALLCLOCK_TIME=01:00:00

# No archiving during dev
./xmlchange DOUT_S=FALSE

# PE layout: 256 cores (confirmed working at 45 SYPD on Derecho)
# ATM/CPL/ICE on PEs 0-127, OCN on PEs 128-255
./xmlchange NTASKS_ATM=128,NTASKS_CPL=128,NTASKS_ICE=128
./xmlchange NTASKS_OCN=128,NTASKS_WAV=128,NTASKS_GLC=128
./xmlchange NTASKS_ROF=128,NTASKS_LND=128
./xmlchange ROOTPE_OCN=128

# =============================================================================
# STEP 3 — case.setup
# =============================================================================
echo "==> Running case.setup..."
./case.setup

# =============================================================================
# STEP 4 — MPI env patches (GPU + MPI coexistence on Derecho)
# =============================================================================
echo "==> Patching env_mach_specific.xml for MPI/GPU compatibility..."
for nameval in \
    "MPICH_GPU_SUPPORT_ENABLED:0" \
    "FI_CXI_DISABLE_HOST_REGISTER:1" \
    "MPICH_SMP_SINGLE_COPY_MODE:NONE"
do
    varname="${nameval%%:*}"
    varval="${nameval##*:}"
    if ! grep -q "name=\"${varname}\"" env_mach_specific.xml; then
        sed -i "s|</environment_variables>|    <env name=\"${varname}\">${varval}</env>\n  </environment_variables>|" \
            env_mach_specific.xml
        echo "    Added ${varname}=${varval}"
    fi
done

# =============================================================================
# STEP 5 — user namelists
# =============================================================================
echo "==> Writing user namelists..."

cat >> user_nl_datm << 'EOF'
datamode = 'CAMULATOR'
EOF

# CICE subcycling: 2x per 6-hr step keeps CFL stable with our MLP winds
cat >> user_nl_cice << 'EOF'
ndtd = 2
EOF

# =============================================================================
# STEP 6 — build
# =============================================================================
if [[ "$1" != "nobuild" ]]; then
    echo "==> Running case.build (~20-40 min)..."
    ./case.build
    echo ""
    echo "==> Build complete."
    echo "    Run directory: ${RUNDIR}"
    echo ""
    echo "Next steps:"
    echo "  1. Copy fake_atm_server.py and output_full/ to ${RUNDIR}"
    echo "  2. Start submit_fake_atm_server.pbs on Casper (GPU node)"
    echo "  3. Wait for camulator_server_ready.flag in ${RUNDIR}"
    echo "  4. Then: cd ${CASE_DIR} && ./case.submit"
else
    echo "==> Skipping build. Run ./case.build manually from ${CASE_DIR}"
fi

echo ""
echo "==> Done. Case at: ${CASE_DIR}"
