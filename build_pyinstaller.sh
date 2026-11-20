#!/bin/sh

pyinstaller --onefile \
            --add-data icc_profiles/Gray-elle-V4-g10.icc:icc_profiles \
            --add-data icc_profiles/Gray-elle-V4-g22.icc:icc_profiles \
            --add-data icc_profiles/Rec2020-elle-V4-g10.icc:icc_profiles \
            --add-data icc_profiles/sRGB-elle-V4-g22.icc:icc_profiles \
            --add-data inverter/styles.qss:inverter \
            --collect-all=imagecodecs \
            inverter/ui.py \
            --name nip
