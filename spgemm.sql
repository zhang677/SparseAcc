SELECT A_CRD0, B_CRD1, SUM(A_VAL * B_VAL)
FROM A INNER JOIN B ON A_CRD1 = B_CRD0
GROUP BY A_CRD0, B_CRD1
