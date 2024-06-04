#!/bin/bash

# proofString="TestCornerCaseFineConvergence"
proofString="TOTEST"
if [ "$#" -eq 2 ]; then
  proofString="$2"
fi

if [ "$#" -ne 2 ] && [ "$#" -ne 1 ]; then
    echo "Usage: $0 <arg1> <arg2>"
    echo "       arg1 is a wcnf file."
    echo "       arg2 is the string which is searched in the proofFile."
    exit 42
fi

wcnfFile="$1"
decompressed=0
# Check if the file exists
if [ -f "$wcnfFile" ]; then
    # Check if it's an XZ compressed file
    if file "$wcnfFile" | grep -q "XZ compressed data"; then
        # Remove the .xz extension for the output file
        output_path="/tmp/$(basename "$wcnfFile" .xz)"

        # Decompress the file
        xz -dkc "$wcnfFile" > "$output_path"
        wcnfFile="$output_path"
        decompressed=1

        echo "File decompressed to $wcnfFile"
    fi
else
    echo "The file does not exist."
    exit 43
fi


dirname="$(dirname "$(realpath "$0")")"
currentdir="$(pwd)"
#echo "Directory: $dirname, $currentdir"

randomNumber=$(od -An -N8 -tu8 /dev/urandom | tr -d '[:space:]')
proofFile="/tmp/proof_$randomNumber.pbp"
veripbOutput="/tmp/veripb_output_$randomNumber.pblog"
# shellcheck source=certified-cgss/certified-cgss/bin/activate

cd "$dirname" || {
    echo "Error in Pacose call script!"
    exit 44
}

echo "c Start Pacose on file $wcnfFile: "
/usr/local/scratch/paxiant/gitlab/PacoseMaxSATSolver-Certified/bin/Pacose --encoding dgpw --proofFile "$proofFile" "$wcnfFile"
PacoseStatus=$?

grep -E "$proofString|SHOULDNEVERHAPPEN" "$proofFile"
proofFileReturnValue=$?
ProofStatus=0
if [ $proofFileReturnValue -ne 1 ]; then
    ProofStatus=99
fi

echo "c Start VeriPB: "
# veripb --wcnf --forceCheckDeletion "$wcnfFile" "$proofFile" | tee "$veripbOutput" | sed 's/^/c /' | grep "c Verification succeeded."
veripb --wcnf "$wcnfFile" "$proofFile" | tee "$veripbOutput" | sed 's/^/c /' | grep "c Verification succeeded."
#veripb --wcnf "$wcnfFile" "$proofFile" | tee "$veripbOutput" | sed 's/^/c /' | grep "c Hint: ("
piperv=("${PIPESTATUS[@]}")
VPBStatus="${piperv[0]}"
GrepStatus="${piperv[3]}"

# Calculate inverted status, to find the Hint -- only necessary if grep shouldn't find the string
#if [ $GrepStatus -eq 0 ]; then
#    GrepStatus=1
#else
#    GrepStatus=0
#fi

sed 's/^/c /' "$veripbOutput"

cd "$currentdir" || {
    echo "Error in Pacose call script!"
    exit 1
}

rv=$PacoseStatus
if [[ $PacoseStatus -eq 20 || $PacoseStatus -eq 10 ]]; then
    PacStat=0
fi


# Calculate the return value
rv=$(((PacStat + 30 * VPBStatus + 10 * GrepStatus + ProofStatus) % 254))

# Check if any variable is non-zero and the result is 0
if [[ $rv -eq 0 && ($PacStat -ne 0 || $VPBStatus -ne 0 || $GrepStatus -ne 0 || $ProofStatus -ne 0) ]]; then
    rv=255 # Set to a non-zero value
fi

if [[ $rv -eq 124 || $rv -eq 137 ]]; then
    rv=254 # Set to a non-zero value
fi

rm -f "$proofFile" "$veripbOutput"
#echo "$proofFile $veripbOutput"
find /tmp/ -name "*.pbp" -mmin +120 -delete 2>/dev/null
find /tmp/ -name "*.pblog" -mmin +240 -delete 2>/dev/null

# THIS HAS TO BE REMOVED FOR THE DELTA DEBUGGING!!!!
# find /tmp/ -name "*.wcnf" -mmin +90 -delete 2>/dev/null

echo "c"
echo "c $rv (return value) = ($PacStat (PacoseStatus) + 30 * $VPBStatus (PBStatus) + 10 * $GrepStatus (GrepStatus) + $ProofStatus (ProofStatus)) % 256"

if [ "$decompressed" == "1" ]; then
    rm -f "$wcnfFile"
fi

# to give the right return value as in the new MSE rules
if [[ $rv -eq 0 ]]; then
    rv=$PacoseStatus
fi
exit $rv
