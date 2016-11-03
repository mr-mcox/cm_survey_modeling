SELECT r.person_id, question_code, survey_code, response
FROM cm_response r
INNER JOIN cm_survey_question sq ON r.survey_question_id = sq.survey_question_id
INNER JOIN cm_survey s ON sq.survey_id = s.survey_id
INNER JOIN cm_question q ON sq.question_id = q.question_id
WHERE survey_code IN (
	'09EIS',
	'10EIS',
	'11EIS',
	'12EIS',
	'13EIS',
	'14EIS',
	'15EIS',
	'16EIS',
	'1617F8W',
	'1516EYS',
	'1516MYS',
	'1516F8W',
	'1415EYS',
	'1415MYS',
	'1415F8W',
	'1314EYS',
	'1314MYS',
	'1314F8W',
	'1213EYS',
	'1213MYS',
	'1213F8W',
	'1112EYS',
	'1112MYS',
	'1112F8W',
	'1011EYS',
	'1011MYS',
	'1011R0',
	'0910EYS',
	'0910MYS',
	'0809EYS'
	) AND question_code IN (
	'CSI1',
	'CSI2',
	'CSI3',
	'CSI4',
	'CSI5',
	'CSI6',
	'CSI7',
	'CSI8',
	'CSI10',
	'CSI12',
	'Culture1',
	'CSI',
	'CSI9'
	)
