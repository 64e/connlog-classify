#!/bin/bash
set -eu

maxsize=${1:-"10M"}

find dataset -type f -name conn.log.labeled -size -${maxsize} -exec sed -i 's/   /\t/g' {} \;
