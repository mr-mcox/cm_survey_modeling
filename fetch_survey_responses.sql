SELECT r.person_id, question_code, survey_code, response, group_name as 'Corps'
FROM cm_response r
INNER JOIN cm_survey_question sq ON r.survey_question_id = sq.survey_question_id
INNER JOIN cm_survey s ON sq.survey_id = s.survey_id
INNER JOIN cm_question q ON sq.question_id = q.question_id
LEFT OUTER JOIN cm_groupings gi ON r.person_id = gi.person_id AND s.survey_id = gi.survey_id
INNER JOIN cm_groups g ON gi.group_id = g.group_id
INNER JOIN cm_grouping_category gc ON g.grouping_category_id = gc.grouping_category_id
WHERE grouping_category = 'Corps' AND
survey_code IN (
	'09EIS',
	'10EIS',
	'11EIS',
	'12EIS',
	'13EIS',
	'14EIS',
	'15EIS',
	'16EIS',
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
