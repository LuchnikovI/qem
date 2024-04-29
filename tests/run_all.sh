#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

for test in "${script_dir}"/*.py
do
    if [[ "${test}" != *utils.py ]]
    then
        $test
    fi
done