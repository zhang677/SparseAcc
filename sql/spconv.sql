PRAGMA kernel_size = 3;
SELECT A1.EDGE, SUM(A1.A_VAL * B_VAL)
FROM A AS A1 CROSS JOIN A AS A2
     CROSS JOIN B
WHERE ABS(A1.A_CRD0 - A2.A_CRD0) <= kernel_size AND ABS(A1.A_CRD1 - A2.A_CRD1) <= kernel_size AND B_CRD0 = A1.A_CRD0 - A2.A_CRD0 AND B_CRD1 = A1.A_CRD1 - A2.A_CRD1
GROUP BY (A1.EDGE)