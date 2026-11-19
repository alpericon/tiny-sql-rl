# v3-400: the 28 full-dev questions rescued by the repair turn (hand-read list)

From results/baseline_Qwen__Qwen3-1.7B_v3-checkpoint-400_repair.records.jsonl; the reading is in
results/week5_results.md.

## 1. concert_singer: Show the stadium names without any concert.

- first error: `sql_error: no such column: sc.Stadium_ID`
- first SQL: `SELECT s.Name FROM stadium s WHERE s.Stadium_ID NOT IN ( SELECT sc.Stadium_ID FROM singer_in_concert sc JOIN concert c ON sc.concert_ID = c.concert_ID );`
- repaired SQL: `SELECT s.Name FROM stadium s WHERE s.Stadium_ID NOT IN ( SELECT c.Stadium_ID FROM concert c JOIN singer_in_concert sc ON c.concert_ID = sc.concert_ID );`
- gold: `SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)`

## 2. concert_singer: Find the number of concerts happened in the stadium with the highest capacity .

- first error: `sql_error: near "FROM": syntax error`
- first SQL: `SELECT COUNT(*) AS num_concerts FROM concert JOIN singer_in_concert ON concert.concert_ID = singer_in_concert.concert_ID JOIN singer ON singer_in_concert.Singer_ID = singer.Singer_ID WHERE stadium_ID = (SELECT Stadium_ID FROM stadium WHERE Capacity = (SELECT MAX(Capacity FROM stadium)));`
- repaired SQL: `SELECT COUNT(*) AS num_concerts FROM concert JOIN singer_in_concert ON concert.concert_ID = singer_in_concert.concert_ID JOIN singer ON singer_in_concert.Singer_ID = singer.Singer_ID JOIN stadium ON concert.Stadium_ID = stadium.Stadium_ID WHERE stadium.Capacity = (SELECT MAX(Capacity) FROM stadium);`
- gold: `select count(*) from concert where stadium_id = (select stadium_id from stadium order by capacity desc limit 1)`

## 3. pets_1: How many dog pets are raised by female students?

- first error: `sql_error: no such column: Student.Sex`
- first SQL: `SELECT COUNT(*) FROM Has_Pet JOIN Pets ON Has_Pet.PetID = Pets.PetID WHERE Pets.PetType = 'dog' AND Student.Sex = 'F' AND Student.StuID = Has_Pet.StuID;`
- repaired SQL: `SELECT COUNT(*) FROM Has_Pet JOIN Pets ON Has_Pet.PetID = Pets.PetID JOIN Student ON Has_Pet.StuID = Student.StuID WHERE Pets.PetType = 'dog' AND Student.Sex = 'F';`
- gold: `SELECT count(*) FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid = T2.stuid JOIN pets AS T3 ON T2.petid = T3.petid WHERE T1.sex = 'F' AND T3.pettype = 'dog'`

## 4. car_1: Find the model of the car whose weight is below the average weight.

- first error: `sql_error: no such column: c.Model`
- first SQL: `SELECT c.Model FROM cars_data c WHERE c.Weight < (SELECT AVG(Weight) FROM cars_data)`
- repaired SQL: `SELECT n.Model FROM cars_data c JOIN car_names n ON c.Id = n.MakeId WHERE c.Weight < (SELECT AVG(Weight) FROM cars_data)`
- gold: `SELECT T1.model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId = T2.Id WHERE T2.Weight < (SELECT avg(Weight) FROM CARS_DATA)`

## 5. car_1: What is the model for the car with a weight smaller than the average?

- first error: `sql_error: no such column: c.Model`
- first SQL: `SELECT c."Model" FROM "cars_data" c WHERE c."Weight" < (SELECT AVG("Weight") FROM "cars_data")`
- repaired SQL: `SELECT n."Model" FROM "cars_data" c JOIN "car_names" n ON c."Id" = n."MakeId" WHERE c."Weight" < (SELECT AVG("Weight") FROM "cars_data")`
- gold: `SELECT T1.model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId = T2.Id WHERE T2.Weight < (SELECT avg(Weight) FROM CARS_DATA)`

## 6. car_1: What is the average weight of cars each year?

- first error: `sql_error: no such function: YEAR`
- first SQL: `SELECT YEAR("cars_data"."Year") AS Year, AVG("cars_data"."Weight") AS AverageWeight FROM "cars_data" GROUP BY YEAR("cars_data"."Year") ORDER BY YEAR("cars_data"."Year");`
- repaired SQL: `SELECT c."Year" AS Year, AVG(c."Weight") AS AverageWeight FROM "cars_data" c GROUP BY c."Year" ORDER BY c."Year";`
- gold: `SELECT avg(Weight) , YEAR FROM CARS_DATA GROUP BY YEAR;`

## 7. flight_2: Count the number of United Airlines flights that arrive in Aberdeen.

- first error: `sql_error: no such column: airports.Airline`
- first SQL: `SELECT COUNT(*) FROM flights JOIN airports ON flights.DestAirport = airports.AirportCode WHERE airports.AirportName LIKE '%Aberdeen%' AND airports.Airline = 1;`
- repaired SQL: `SELECT COUNT(*) FROM flights JOIN airports ON flights.DestAirport = airports.AirportCode WHERE airports.AirportName LIKE '%Aberdeen%' AND airports.AirportCode = (SELECT AirportCode FROM airports WHERE AirportName LIKE '%Aberdeen%');`
- gold: `SELECT count(*) FROM FLIGHTS AS T1 JOIN AIRPORTS AS T2 ON T1.DestAirport = T2.AirportCode JOIN AIRLINES AS T3 ON T3.uid = T1.Airline WHERE T2.City = "Aberdeen" AND T3.Airline = "United Airlines"`

## 8. employee_hire_evaluation: Which cities do more than one employee under age 30 come from?

- first error: `sql_error: no such column: s.City`
- first SQL: `SELECT s.City FROM employee e JOIN shop s ON e.Employee_ID = hiring.Employee_ID WHERE e.Age < 30 GROUP BY s.City HAVING COUNT(DISTINCT e.Employee_ID) > 1;`
- repaired SQL: `SELECT e.City FROM employee e JOIN hiring h ON e.Employee_ID = h.Employee_ID WHERE e.Age < 30 GROUP BY e.City HAVING COUNT(DISTINCT e.Employee_ID) > 1;`
- gold: `SELECT city FROM employee WHERE age < 30 GROUP BY city HAVING count(*) > 1`

## 9. cre_Doc_Template_Mgt: What are the different template type codes, and how many documents use each type?

- first error: `sql_error: no such column: D.D.Document_ID`
- first SQL: `SELECT rt.Template_Type_Code, COUNT(D.D.Document_ID) AS Document_Count FROM Ref_Template_Types rt JOIN Templates T ON rt.Template_Type_Code = T.Template_Type_Code JOIN Documents D ON T.Template_ID = D.Template_ID GROUP BY rt.Template_Type_Code;`
- repaired SQL: `SELECT rt.Template_Type_Code, COUNT(D.TEMPLATE_ID) AS Document_Count FROM Ref_Template_Types rt JOIN Templates T ON rt.Template_Type_Code = T.Template_Type_Code JOIN Documents D ON T.Template_ID = D.Template_ID GROUP BY rt.Template_Type_Code;`
- gold: `SELECT T1.template_type_code , count(*) FROM Templates AS T1 JOIN Documents AS T2 ON T1.template_id = T2.template_id GROUP BY T1.template_type_code`

## 10. cre_Doc_Template_Mgt: Return the ids corresponding to templates with the description 'Presentation'.

- first error: `sql_error: no such column: Template_Type_Description`
- first SQL: `SELECT Template_ID FROM Templates WHERE Template_Type_Description = 'Presentation';`
- repaired SQL: `SELECT Template_ID FROM Templates WHERE Template_Type_Code IN (SELECT Template_Type_Code FROM Ref_Template_Types WHERE Template_Type_Description = 'Presentation');`
- gold: `SELECT T2.template_id FROM Ref_template_types AS T1 JOIN Templates AS T2 ON T1.template_type_code = T2.template_type_code WHERE T1.template_type_description = "Presentation"`

## 11. course_teach: What is the name of each teacher and what course they teach?

- first error: `sql_error: no such column: course_arrange.Teacher_ID`
- first SQL: `SELECT t.Name, c.Course FROM teacher t JOIN course c ON t.Teacher_ID = course_arrange.Teacher_ID`
- repaired SQL: `SELECT t.Name, c.Course FROM teacher t JOIN course_arrange ca ON t.Teacher_ID = ca.Teacher_ID JOIN course c ON ca.Course_ID = c.Course_ID`
- gold: `SELECT T3.Name , T2.Course FROM course_arrange AS T1 JOIN course AS T2 ON T1.Course_ID = T2.Course_ID JOIN teacher AS T3 ON T1.Teacher_ID = T3.Teacher_ID`

## 12. museum_visit: What are the id and name of the museum visited most times?

- first error: `sql_error: no such column: v.ID`
- first SQL: `SELECT v."ID", v."Name" FROM "museum" m JOIN "visit" v ON m."Museum_ID" = v."Museum_ID" GROUP BY m."Museum_ID" ORDER BY COUNT(v."visitor_ID") DESC LIMIT 1;`
- repaired SQL: `SELECT m."Museum_ID", m."Name" FROM "museum" m JOIN "visit" v ON m."Museum_ID" = v."Museum_ID" GROUP BY m."Museum_ID" ORDER BY COUNT(v."visitor_ID") DESC LIMIT 1;`
- gold: `SELECT t2.Museum_ID , t1.name FROM museum AS t1 JOIN visit AS t2 ON t1.Museum_ID = t2.Museum_ID GROUP BY t2.Museum_ID ORDER BY count(*) DESC LIMIT 1`

## 13. wta_1: find the number of distinct country codes of all players.

- first error: `sql_error: near "FROM": syntax error`
- first SQL: `SELECT COUNT(DISTINCT "country_code" FROM players);`
- repaired SQL: `SELECT COUNT(DISTINCT country_code) FROM players;`
- gold: `SELECT count(DISTINCT country_code) FROM players`

## 14. student_transcripts_tracking: What is the zip code of the address in the city Port Chelsea?

- first error: `sql_error: no such column: address.zip_postcode`
- first SQL: `SELECT DISTINCT address.zip_postcode FROM Addresses WHERE Addresses.city = 'Port Chelsea';`
- repaired SQL: `SELECT DISTINCT zip_postcode FROM Addresses WHERE city = 'Port Chelsea';`
- gold: `SELECT zip_postcode FROM Addresses WHERE city = 'Port Chelsea'`

## 15. student_transcripts_tracking: What is the description for the section named h?

- first error: `no ```sql block`
- first SQL: `(no sql block)`
- repaired SQL: `SELECT section_description FROM Sections WHERE section_name = 'h';`
- gold: `SELECT section_description FROM Sections WHERE section_name = 'h'`

## 16. tvshow: What is the name and directors of all the cartoons that are ordered by air date?

- first error: `sql_error: no such column: t.Directors`
- first SQL: `SELECT t.Title, t.Directors FROM Cartoon t ORDER BY t.Original_air_date ASC;`
- repaired SQL: `SELECT Title, Directed_by FROM Cartoon ORDER BY Original_air_date ASC;`
- gold: `SELECT title , Directed_by FROM Cartoon ORDER BY Original_air_date`

## 17. poker_player: What are the names of poker players?

- first error: `sql_error: no such column: p.Name`
- first SQL: `SELECT p.Name FROM poker_player p`
- repaired SQL: `SELECT people.Name FROM poker_player JOIN people ON poker_player.People_ID = people.People_ID`
- gold: `SELECT T1.Name FROM people AS T1 JOIN poker_player AS T2 ON T1.People_ID = T2.People_ID`

## 18. poker_player: Return the names of all the poker players.

- first error: `sql_error: no such column: p.Name`
- first SQL: `SELECT p.Name FROM poker_player p;`
- repaired SQL: `SELECT people.Name FROM poker_player JOIN people ON poker_player.People_ID = people.People_ID;`
- gold: `SELECT T1.Name FROM people AS T1 JOIN poker_player AS T2 ON T1.People_ID = T2.People_ID`

## 19. poker_player: List the names of poker players ordered by the final tables made in ascending order.

- first error: `sql_error: no such column: p.Name`
- first SQL: `SELECT p.Name FROM poker_player p ORDER BY p.Final_Table_Made ASC;`
- repaired SQL: `SELECT people.Name FROM poker_player JOIN people ON poker_player.People_ID = people.People_ID ORDER BY poker_player.Final_Table_Made ASC;`
- gold: `SELECT T1.Name FROM people AS T1 JOIN poker_player AS T2 ON T1.People_ID = T2.People_ID ORDER BY T2.Final_Table_Made`

## 20. poker_player: What are the names of poker players, ordered ascending by the number of final tables they have made?

- first error: `sql_error: no such column: p.Name`
- first SQL: `SELECT p.Name FROM poker_player p JOIN people pe ON p.People_ID = pe.People_ID ORDER BY p.Final_Table_Made ASC;`
- repaired SQL: `SELECT pe.Name FROM poker_player p JOIN people pe ON p.People_ID = pe.People_ID ORDER BY p.Final_Table_Made ASC;`
- gold: `SELECT T1.Name FROM people AS T1 JOIN poker_player AS T2 ON T1.People_ID = T2.People_ID ORDER BY T2.Final_Table_Made`

## 21. poker_player: What is the birth date of the poker player with the lowest earnings?

- first error: `sql_error: no such column: p.birth_date`
- first SQL: `SELECT p.birth_date FROM poker_player p JOIN people pe ON p.People_ID = pe.People_ID WHERE p.Earnings = (SELECT MIN(Earnings) FROM poker_player) ORDER BY p.Earnings ASC LIMIT 1;`
- repaired SQL: `SELECT pe.Birth_Date FROM poker_player p JOIN people pe ON p.People_ID = pe.People_ID WHERE p.Earnings = (SELECT MIN(Earnings) FROM poker_player) ORDER BY p.Earnings ASC LIMIT 1;`
- gold: `SELECT T1.Birth_Date FROM people AS T1 JOIN poker_player AS T2 ON T1.People_ID = T2.People_ID ORDER BY T2.Earnings ASC LIMIT 1`

## 22. poker_player: Return the birth date of the poker player with the lowest earnings.

- first error: `sql_error: no such column: p.birth_date`
- first SQL: `SELECT p.birth_date FROM poker_player p JOIN people pe ON p.People_ID = pe.People_ID WHERE p.Earnings = (SELECT MIN(Earnings) FROM poker_player) LIMIT 1;`
- repaired SQL: `SELECT pe.Birth_Date FROM poker_player p JOIN people pe ON p.People_ID = pe.People_ID WHERE p.Earnings = (SELECT MIN(Earnings) FROM poker_player) LIMIT 1;`
- gold: `SELECT T1.Birth_Date FROM people AS T1 JOIN poker_player AS T2 ON T1.People_ID = T2.People_ID ORDER BY T2.Earnings ASC LIMIT 1`

## 23. poker_player: Give average earnings of poker players who are taller than 200.

- first error: `sql_error: no such column: Height`
- first SQL: `SELECT AVG(Earnings) AS Average_Earnings FROM poker_player WHERE Height > 200;`
- repaired SQL: `SELECT AVG(Earnings) AS Average_Earnings FROM poker_player JOIN people ON poker_player.People_ID = people.People_ID WHERE people.Height > 200;`
- gold: `SELECT avg(T2.Earnings) FROM people AS T1 JOIN poker_player AS T2 ON T1.People_ID = T2.People_ID WHERE T1.Height > 200`

## 24. poker_player: Return the names of poker players sorted by their earnings descending.

- first error: `sql_error: no such column: p.Name`
- first SQL: `SELECT p.Name FROM poker_player p JOIN people pe ON p.People_ID = pe.People_ID ORDER BY p.Earnings DESC;`
- repaired SQL: `SELECT pe.Name FROM poker_player p JOIN people pe ON p.People_ID = pe.People_ID ORDER BY p.Earnings DESC;`
- gold: `SELECT T1.Name FROM people AS T1 JOIN poker_player AS T2 ON T1.People_ID = T2.People_ID ORDER BY T2.Earnings DESC`

## 25. world_1: Which region is the city Kabul located in?

- first error: `sql_error: no such column: c.Region`
- first SQL: `SELECT c.Region FROM city c JOIN country c2 ON c.CountryCode = c2.Code WHERE c.Name = 'Kabul';`
- repaired SQL: `SELECT c2.Region FROM city c JOIN country c2 ON c.CountryCode = c2.Code WHERE c.Name = 'Kabul';`
- gold: `SELECT Region FROM country AS T1 JOIN city AS T2 ON T1.Code = T2.CountryCode WHERE T2.Name = "Kabul"`

## 26. orchestra: Return the average attendance across all shows.

- first error: `sql_error: no such column: Share`
- first SQL: `SELECT AVG(Share) AS average_attendance FROM show`
- repaired SQL: `SELECT AVG(Attendance) AS average_attendance FROM show`
- gold: `SELECT avg(Attendance) FROM SHOW`

## 27. orchestra: What is the name of the conductor who has conducted the most orchestras?

- first error: `sql_error: no such column: p.Orchestra_ID`
- first SQL: `SELECT p.Name FROM conductor p GROUP BY p.Conductor_ID ORDER BY COUNT(p.Orchestra_ID) DESC LIMIT 1;`
- repaired SQL: `SELECT c.Name FROM conductor c JOIN orchestra o ON c.Conductor_ID = o.Conductor_ID GROUP BY c.Conductor_ID ORDER BY COUNT(o.Orchestra_ID) DESC LIMIT 1;`
- gold: `SELECT T1.Name FROM conductor AS T1 JOIN orchestra AS T2 ON T1.Conductor_ID = T2.Conductor_ID GROUP BY T2.Conductor_ID ORDER BY COUNT(*) DESC LIMIT 1`

## 28. dog_kennels: What is the average age of all the dogs?

- first error: `no ```sql block`
- first SQL: `(no sql block)`
- repaired SQL: `SELECT AVG(age) AS average_age FROM Dogs;`
- gold: `SELECT avg(age) FROM Dogs`
