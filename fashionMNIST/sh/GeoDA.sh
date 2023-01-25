query1=300
query2=200


# adv_train, proxy
python src/GeoDA.py --AK=robust_white --query=${query1}  --output_path=output/GeoDA/robust_white > output/GeoDA/robust_white/query${query1}.txt
python src/GeoDA.py --AK=robust_white --query=${query2}  --output_path=output/GeoDA/robust_white > output/GeoDA/robust_white/query${query2}.txt

