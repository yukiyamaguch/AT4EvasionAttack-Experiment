query1=5
query2=10

# adv_train, proxy
python src/HopSkipJumpAttack.py --AK=white --query=${query1}  --output_path=output/HopSkipJumpAttack/white > output/HopSkipJumpAttack/white/query${query1}.txt
python src/HopSkipJumpAttack.py --AK=white --query=${query2}  --output_path=output/HopSkipJumpAttack/white > output/HopSkipJumpAttack/white/query${query2}.txt

