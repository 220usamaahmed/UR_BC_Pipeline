#!/bin/bash

JOINT_STATE=$(ros2 topic echo /joint_states --once)

mapfile -t JOINT_NAMES < <(
    echo "$JOINT_STATE" |
        sed -n '/^name:/,/^position:/p' |
        sed -n '2,$p' |
        sed '$d' |
        sed "s/^[[:space:]]*-[[:space:]]*//; s/['\"]//g"
)

mapfile -t JOINT_POSITIONS < <(
    echo "$JOINT_STATE" |
        sed -n '/^position:/,/^velocity:/p' |
        sed -n '2,$p' |
        sed '$d' |
        sed 's/^[[:space:]]*-[[:space:]]*//'
)

declare -A POSITION_BY_JOINT
for i in "${!JOINT_NAMES[@]}"; do
    POSITION_BY_JOINT["${JOINT_NAMES[$i]}"]="${JOINT_POSITIONS[$i]}"
done

JOINT_ORDER=(
    "shoulder_lift_joint"
    "elbow_joint"
    "wrist_1_joint"
    "wrist_2_joint"
    "wrist_3_joint"
    "shoulder_pan_joint"
)

ORDERED_POSITIONS=()
for joint in "${JOINT_ORDER[@]}"; do
    if [[ -z "${POSITION_BY_JOINT[$joint]+_}" ]]; then
        echo "Error: joint '$joint' was not found in /joint_states" >&2
        exit 1
    fi
    ORDERED_POSITIONS+=("${POSITION_BY_JOINT[$joint]}")
done

echo "Raw Radians: ${ORDERED_POSITIONS[*]}"

# Convert to degrees and format as a Python list.
DEGREES_ARRAY=$(printf '%s\n' "${ORDERED_POSITIONS[*]}" | awk '{
    printf "[";
    for (i=1; i<=NF; i++) {
        # awk handles scientific notation automatically in math
        deg = $i * 180 / 3.141592653589793;
        printf "%.2f%s", deg, (i==NF ? "" : ", ");
    }
    print "]";
}')

echo "UR3e Joint Angles (Degrees):"
echo "$DEGREES_ARRAY"
