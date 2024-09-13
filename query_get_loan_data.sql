SELECT age, income, dependents, has_property, has_car, credit_score,
       job_tenure, has_education, loan_amount, dateDiff('day', loan_start, loan_deadline) AS loan_period, 
       if(dateDiff('day', loan_deadline, loan_payed) < 0, 0, dateDiff('day', loan_deadline, loan_payed)) AS delay_days
FROM default.loan_delay_days
ORDER BY id;