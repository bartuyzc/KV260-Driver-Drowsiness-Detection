#!/bin/bash
# power_monitor.sh
# ----------------
# Periodically reads Power Utilization and PL Temperature values
# from xmutil platformstats output and logs them into a CSV file.
#
# Usage:
#   chmod +x power_monitor.sh
#   ./power_monitor.sh                         # default: 1s interval, infinite
#   ./power_monitor.sh --interval 2            # 2 seconds interval
#   ./power_monitor.sh --interval 1 --duration 60  # 60 seconds
#
# Stop with: Ctrl+C

INTERVAL=1
DURATION=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --interval)  INTERVAL="$2"; shift 2 ;;
        --duration)  DURATION="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTFILE="reports/power_log_${TIMESTAMP}.csv"
SUMMARY="reports/power_summary_${TIMESTAMP}.txt"

echo ">> Power Analyzer Starting..."
echo ">> Interval : ${INTERVAL} seconds"
echo ">> Duration : $([ $DURATION -eq 0 ] && echo 'Run until Ctrl+C' || echo "${DURATION} seconds")"
echo ">> CSV log  : ${OUTFILE}"
echo ">> Summary  : ${SUMMARY}"
echo ""

# CSV header
echo "timestamp,elapsed_s,som_power_mW,som_current_mA,som_voltage_mV,pl_temp_C" > "$OUTFILE"

START=$(date +%s)
SAMPLE=0

printf "%-10s %6s %15s %15s %15s %12s\n" \
    "Time" "Elapsed" "Power(mW)" "Current(mA)" "Voltage(mV)" "PL Temp(C)"
printf "%-10s %6s %15s %15s %15s %12s\n" \
    "----------" "------" "---------------" "---------------" "---------------" "------------"

cleanup() {
    echo ""
    echo ">> Stopped. Total ${SAMPLE} measurements."
    echo ">> CSV saved to  : ${OUTFILE}"
    echo ">> Summary saved : ${SUMMARY}"
    echo ""

    SUMMARY_CONTENT=$(awk -F',' 'NR>1 {
        sp+=$3; sc+=$4; sv+=$5; pt+=$6; n++
        if($3>max_p) max_p=$3
        if($3<min_p || min_p==0) min_p=$3
        if($6>max_t) max_t=$6
    } END {
        printf "  Total Samples     : %d\n", n
        printf "  Average Power     : %.1f mW\n", sp/n
        printf "  Min / Max Power   : %.1f / %.1f mW\n", min_p, max_p
        printf "  Average Current   : %.1f mA\n", sc/n
        printf "  Average Voltage   : %.1f mV\n", sv/n
        printf "  Average PL Temp   : %.1f C\n",  pt/n
        printf "  Max PL Temp       : %.1f C\n",  max_t
    }' "$OUTFILE")

    echo "========================================="
    echo "  SUMMARY"
    echo "========================================="
    echo "$SUMMARY_CONTENT"
    echo "========================================="

    # Write summary txt
    {
        echo "========================================="
        echo "  POWER MONITOR SUMMARY"
        echo "========================================="
        echo "  Date/Time         : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  Interval          : ${INTERVAL}s"
        echo "  CSV log           : ${OUTFILE}"
        echo "-----------------------------------------"
        echo "$SUMMARY_CONTENT"
        echo "========================================="
    } > "$SUMMARY"

    exit 0
}
trap cleanup SIGINT SIGTERM

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    TS=$(date +"%H:%M:%S")

    RAW=$(xmutil xlnx_platformstats 2>/dev/null)

    if [ -z "$RAW" ]; then
        echo "[WARNING] Failed to read platformstats..."
        sleep "$INTERVAL"
        continue
    fi

    # Parse power lines
    SOM_PWR=$(echo "$RAW" | grep "SOM total power"    | grep -oE '[0-9]+' | tail -1)
    SOM_CUR=$(echo "$RAW" | grep "SOM total current"  | grep -oE '[0-9]+' | tail -1)
    SOM_VOL=$(echo "$RAW" | grep "SOM total voltage"  | grep -oE '[0-9]+' | tail -1)
    PL_TEMP=$(echo "$RAW" | grep "PL temperature "    | grep -oE '[0-9]+' | tail -1)

    # Default values
    SOM_PWR=${SOM_PWR:-0}
    SOM_CUR=${SOM_CUR:-0}
    SOM_VOL=${SOM_VOL:-0}
    PL_TEMP=${PL_TEMP:-0}

    SAMPLE=$((SAMPLE + 1))

    # Terminal output
    printf "%-10s %6s %15s %15s %15s %12s\n" \
        "$TS" "${ELAPSED}s" "${SOM_PWR} mW" "${SOM_CUR} mA" "${SOM_VOL} mV" "${PL_TEMP} C"

    # CSV log
    echo "${TS},${ELAPSED},${SOM_PWR},${SOM_CUR},${SOM_VOL},${PL_TEMP}" >> "$OUTFILE"

    if [ "$DURATION" -gt 0 ] && [ "$ELAPSED" -ge "$DURATION" ]; then
        cleanup
    fi

    sleep "$INTERVAL"
done
