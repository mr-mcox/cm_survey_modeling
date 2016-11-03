SELECT gi.person_id, survey_code, grouping_category, group_name
FROM cm_survey s
INNER JOIN cm_groupings gi ON s.survey_id = gi.survey_id
INNER JOIN cm_groups g ON gi.group_id = g.group_id
INNER JOIN cm_grouping_category gc ON g.grouping_category_id = gc.grouping_category_id
WHERE (grouping_category = 'Corps' OR grouping_category = 'Region') AND
survey_code IN (
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
	) 