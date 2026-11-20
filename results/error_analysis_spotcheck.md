# Week 9 spot check (E3): label these 20 without looking at my labels

For each question, write **one primary label** (and optionally one secondary) from the list below:
the first thing that would have to change for the predicted SQL to be right.

| Label | Meaning |
|---|---|
| `schema-link` | wrong table or column for a concept in the question |
| `join` | wrong or missing join path, or a join that duplicates rows |
| `condition` | missing, extra or wrong WHERE / HAVING condition (not a literal issue) |
| `literal` | right condition, wrong value or value format |
| `aggregation` | wrong aggregate, GROUP BY, DISTINCT or count semantics |
| `order-limit` | wrong ORDER BY direction or column, a missing LIMIT, or tie handling |
| `nesting-setop` | wrong subquery, INTERSECT / EXCEPT / UNION, or NOT IN logic |
| `output-shape` | right rows, wrong columns: extra, missing or reordered columns |
| `gold-issue` | the gold query is wrong, or the question is ambiguous and the prediction is a valid reading |
| `other` | none of the above |

Please don't open `results/error_analysis_labels.jsonl` until you're done. Reply with lines like `7: join`.

## 7 — car_1 (medium)

**Question:** What is the full name of each car maker, along with its id and how many models it produces?

```sql
-- gold
SELECT T1.FullName ,  T1.Id ,  count(*) FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker GROUP BY T1.Id;
-- predicted
SELECT cm.Id AS Id, cm.Maker AS Maker, cm.FullName AS FullName, COUNT(*) AS ModelCount FROM car_makers cm GROUP BY cm.Id, cm.Maker;
```

- gold result: 22 rows — American Motor Company | 1 | 1; Volkswagen | 2 | 2; BMW | 3 | 1
- predicted result: 22 rows — 1 | amc | American Motor Company | 1; 2 | volkswagen | Volkswagen | 1; 3 | bmw | BMW | 1

**Your label:** 

## 13 — car_1 (medium)

**Question:** What are all the makers and models?

```sql
-- gold
SELECT Maker ,  Model FROM MODEL_LIST;
-- predicted
SELECT car_makers.Maker, car_names.Model FROM car_makers JOIN car_names ON car_makers.Id = car_names.MakeId;
```

- gold result: 36 rows — 1 | amc; 2 | audi; 3 | bmw
- predicted result: 22 rows — amc | chevrolet; volkswagen | buick; bmw | plymouth

**Your label:** 

## 16 — car_1 (medium)

**Question:** How many car models were produced by the maker with full name American Motor Company?

```sql
-- gold
SELECT count(*) FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker WHERE T1.FullName  =  'American Motor Company';
-- predicted
SELECT COUNT(*) FROM car_makers WHERE Maker = 'American Motor Company' AND FullName = 'American Motor Company';
```

- gold result: 1 rows — 1
- predicted result: 1 rows — 0

**Your label:** 

## 27 — flight_2 (extra)

**Question:** What is the code of airport that has the highest number of flights?

```sql
-- gold
SELECT T1.AirportCode FROM AIRPORTS AS T1 JOIN FLIGHTS AS T2 ON T1.AirportCode  =  T2.DestAirport OR T1.AirportCode  =  T2.SourceAirport GROUP BY T1.AirportCode ORDER BY count(*) DESC LIMIT 1
-- predicted
SELECT DestAirport FROM flights GROUP BY DestAirport ORDER BY COUNT(*) DESC LIMIT 1;
```

- gold result: 0 rows
- predicted result: 1 rows —  AXG

**Your label:** 

## 28 — flight_2 (extra)

**Question:** What airline serves the most flights?

```sql
-- gold
SELECT T1.Airline FROM AIRLINES AS T1 JOIN FLIGHTS AS T2 ON T1.uid  =  T2.Airline GROUP BY T1.Airline ORDER BY count(*) DESC LIMIT 1
-- predicted
SELECT Airline FROM airlines WHERE Airline IN (SELECT Airline FROM flights GROUP BY Airline ORDER BY COUNT(*) DESC LIMIT 1)
```

- gold result: 1 rows — Virgin America
- predicted result: 0 rows

**Your label:** 

## 31 — cre_Doc_Template_Mgt (easy)

**Question:** Count the number of different templates used for documents.

```sql
-- gold
SELECT count(DISTINCT template_id) FROM Documents
-- predicted
SELECT COUNT(DISTINCT Template_ID) AS Unique_Templates FROM Templates;
```

- gold result: 1 rows — 12
- predicted result: 1 rows — 20

**Your label:** 

## 33 — cre_Doc_Template_Mgt (easy)

**Question:** What are the template ids of any templates used in more than a single document?

```sql
-- gold
SELECT template_id FROM Documents GROUP BY template_id HAVING count(*)  >  1
-- predicted
SELECT t.Template_ID FROM Templates t WHERE t.Template_ID IN ( SELECT d.Document_ID FROM Documents d GROUP BY d.Document_ID HAVING COUNT(*) > 1 );
```

- gold result: 3 rows — 11; 14; 25
- predicted result: 0 rows

**Your label:** 

## 34 — cre_Doc_Template_Mgt (hard)

**Question:** Which template type code has most number of templates?

```sql
-- gold
SELECT template_type_code FROM Templates GROUP BY template_type_code ORDER BY count(*) DESC LIMIT 1
-- predicted
SELECT Template_Type_Code, COUNT(*) as count FROM Ref_Template_Types GROUP BY Template_Type_Code ORDER BY count DESC LIMIT 1;
```

- gold result: 1 rows — PP
- predicted result: 1 rows — AD | 1

**Your label:** 

## 39 — cre_Doc_Template_Mgt (medium)

**Question:** Return the different descriptions for templates that have been used in a document.

```sql
-- gold
SELECT DISTINCT T1.template_type_description FROM Ref_template_types AS T1 JOIN Templates AS T2 ON T1.template_type_code  = T2.template_type_code JOIN Documents AS T3 ON T2.Template_ID  =  T3.template_ID
-- predicted
SELECT Template_Type_Description FROM Templates JOIN Ref_Template_Types ON Templates.Template_Type_Code = Ref_Template_Types.Template_Type_Code
```

- gold result: 4 rows — Presentation; Paper; Book
- predicted result: 20 rows — Paper; Paper; Book

**Your label:** 

## 42 — course_teach (medium)

**Question:** What is the age and hometown of every teacher?

```sql
-- gold
SELECT Age ,  Hometown FROM teacher
-- predicted
SELECT "teacher"."Teacher_ID", "teacher"."Name", "teacher"."Age", "teacher"."Hometown" FROM "teacher";
```

- gold result: 7 rows — 32 | Blackrod Urban District; 29 | Bolton County Borough; 26 | Farnworth Municipal Borough
- predicted result: 7 rows — 1 | Joseph Huts | 32 | Blackrod Urban District; 2 | Gustaaf Deloor | 29 | Bolton County Borough; 3 | Vicente Carretero | 26 | Farnworth Municipal Borough

**Your label:** 

## 45 — museum_visit (extra)

**Question:** What are the id and name of the museum visited most times?

```sql
-- gold
SELECT t2.Museum_ID ,  t1.name FROM museum AS t1 JOIN visit AS t2 ON t1.Museum_ID  =  t2.Museum_ID GROUP BY t2.Museum_ID ORDER BY count(*) DESC LIMIT 1
-- predicted
SELECT v."ID", v."Name" FROM "visitor" v JOIN "visit" s ON v."ID" = s."visitor_ID" JOIN "museum" m ON s."Museum_ID" = m."Museum_ID" GROUP BY v."ID", v."Name" ORDER BY COUNT(s."visitor_ID") DESC LIMIT 1;
```

- gold result: 1 rows — 2 | Capital Plaza Museum
- predicted result: 1 rows — 3 | Arjen Robben

**Your label:** 

## 51 — wta_1 (hard)

**Question:** Find the year that has the most number of matches.

```sql
-- gold
SELECT YEAR FROM matches GROUP BY YEAR ORDER BY count(*) DESC LIMIT 1
-- predicted
SELECT year FROM matches GROUP BY year ORDER BY year DESC LIMIT 1;
```

- gold result: 1 rows — 2016
- predicted result: 1 rows — 2017

**Your label:** 

## 52 — wta_1 (medium)

**Question:** What are the names and ranks of the three youngest winners across all matches?

```sql
-- gold
SELECT DISTINCT winner_name ,  winner_rank FROM matches ORDER BY winner_age LIMIT 3
-- predicted
SELECT p."first_name", p."last_name", r."ranking" FROM players p JOIN rankings r ON p."player_id" = r."player_id" WHERE r."ranking_date" = (SELECT MIN(ranking_date) FROM rankings) ORDER BY p."birth_date" ASC LIMIT 3;
```

- gold result: 3 rows — Madison Keys | 17; Madison Keys | 24; Madison Keys | 16
- predicted result: 1 rows — Brenda | Schultz Mccarthy | 289

**Your label:** 

## 53 — wta_1 (medium)

**Question:** How many different winners both participated in the WTA Championships and were left handed?

```sql
-- gold
SELECT count(DISTINCT winner_name) FROM matches WHERE tourney_name  =  'WTA Championships' AND winner_hand  =  'L'
-- predicted
SELECT COUNT(DISTINCT winner_id) FROM matches WHERE winner_id IN (SELECT player_id FROM players WHERE hand = 'R') AND winner_name IN (SELECT first_name FROM players WHERE hand = 'R')
```

- gold result: 1 rows — 2
- predicted result: 1 rows — 0

**Your label:** 

## 60 — student_transcripts_tracking (extra)

**Question:** Find the kind of program which most number of students are enrolled in?

```sql
-- gold
SELECT T1.degree_summary_name FROM Degree_Programs AS T1 JOIN Student_Enrolment AS T2 ON T1.degree_program_id  =  T2.degree_program_id GROUP BY T1.degree_summary_name ORDER BY count(*) DESC LIMIT 1
-- predicted
SELECT dp.degree_summary_name, COUNT(*) as student_count FROM Degree_Programs dp JOIN Student_Enrolment se ON dp.degree_program_id = se.degree_program_id GROUP BY dp.degree_program_id ORDER BY student_count DESC LIMIT 1;
```

- gold result: 1 rows — Bachelor
- predicted result: 1 rows — Bachelor | 3

**Your label:** 

## 65 — tvshow (medium)

**Question:** What are the package options and the name of the series for the TV Channel that supports high definition TV?

```sql
-- gold
SELECT package_option ,  series_name FROM TV_Channel WHERE hight_definition_TV  =  "yes"
-- predicted
SELECT "Package_Option", "Title" FROM "Cartoon" JOIN "TV_Channel" ON "Cartoon"."Channel" = "TV_Channel"."id" WHERE "TV_Channel"."Hight_definition_TV" = 'yes';
```

- gold result: 1 rows — Sky Famiglia + Sky HD | MTV Live HD
- predicted result: 0 rows

**Your label:** 

## 75 — world_1 (medium)

**Question:** What is the official language spoken in the country whose head of state is Beatrix?

```sql
-- gold
SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.HeadOfState  =  "Beatrix" AND T2.IsOfficial  =  "T"
-- predicted
SELECT l.Language FROM countrylanguage l WHERE l.CountryCode IN ( SELECT c.Code FROM country c WHERE c.HeadOfState = 'Beatrix' );
```

- gold result: 4 rows — Dutch; Dutch; Papiamento
- predicted result: 11 rows — Dutch; English; Papiamento

**Your label:** 

## 86 — orchestra (hard)

**Question:** What are the orchestras that do not have any performances?

```sql
-- gold
SELECT Orchestra FROM orchestra WHERE Orchestra_ID NOT IN (SELECT Orchestra_ID FROM performance)
-- predicted
SELECT Orchestra_ID FROM orchestra WHERE Orchestra_ID NOT IN ( SELECT Orchestra_ID FROM performance )
```

- gold result: 1 rows — San Francisco Symphony Orchestra
- predicted result: 1 rows — 12

**Your label:** 

## 90 — network_1 (medium)

**Question:** Show the names of high schoolers who have at least 3 friends.

```sql
-- gold
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id GROUP BY T1.student_id HAVING count(*)  >=  3
-- predicted
SELECT H.name FROM Highschooler H WHERE H.ID IN ( SELECT F.student_id FROM Friend F WHERE F.student_id IN (SELECT H.ID FROM Highschooler H) );
```

- gold result: 1 rows — Andrew
- predicted result: 14 rows — Haley; Alexis; Jordan

**Your label:** 

## 99 — singer (medium)

**Question:** what are the different names of the singers that have sales more than 300000?

```sql
-- gold
SELECT DISTINCT T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID  =  T2.Singer_ID WHERE T2.Sales  >  300000
-- predicted
SELECT Name FROM singer WHERE Net_Worth_Millions > 30.0
```

- gold result: 3 rows — Liliane Bettencourt; Christy Walton; Iris Fontbona
- predicted result: 0 rows

**Your label:** 
