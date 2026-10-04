#!/usr/bin/env python3
"""One-shot authoring helper: writes the fact-shaped task files for
ab/tasks2/ from the hand-authored passages below (offline build; the
model-driven regeneration loop lives in generate.py)."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

TASKS = [
    # ---------------- medium (~800 chars) ----------------
    dict(
        id="fact_med_01", register="fact", len_class="medium",
        passage=(
            "The steam collier Brixham arrived at Newcastle on 12 October 1884 after a passage of "
            "six days from London, having carried 640 tons of coal. Her master, Captain Elias Ward, "
            "reported strong headwinds off Flamborough Head on the second day, which cost the ship "
            "nearly ten hours. The Brixham was built in 1879 at Middlesbrough and registered at 812 "
            "tons gross. Her engines developed 900 indicated horsepower and consumed twenty-two tons "
            "of coal per day. The owners, Temperley and Carter of London, intended her for the "
            "regular coal run between the Tyne and the capital. On this voyage she carried a crew of "
            "twenty-two men and three officers. Two firemen had been discharged at London for "
            "drunkenness before departure, and their replacements were signed on only hours before "
            "the ship left the dock. The cargo was delivered in good order, and the surveyor's "
            "certificate noted no damage to the hatches or coamings."
        ),
        question="How many tons of coal did the Brixham carry on this voyage?",
        expected_answer="640 tons",
        facts=[
            "The steam collier Brixham arrived at Newcastle on 12 October 1884.",
            "The passage from London took six days.",
            "The Brixham carried 640 tons of coal.",
            "Her master was Captain Elias Ward.",
            "She met strong headwinds off Flamborough Head on the second day.",
            "The headwinds cost the ship nearly ten hours.",
            "The Brixham was built in 1879 at Middlesbrough.",
            "She was registered at 812 tons gross.",
            "Her engines developed 900 indicated horsepower.",
            "The engines consumed twenty-two tons of coal per day.",
            "The owners were Temperley and Carter of London.",
            "The ship carried a crew of twenty-two men and three officers.",
            "Two firemen were discharged at London for drunkenness before departure.",
        ],
    ),
    dict(
        id="fact_med_02", register="fact", len_class="medium",
        passage=(
            "The lighthouse at Cape Verde Point was commissioned on 3 March 1901 and first lit by "
            "keeper Augusto Ferreira, who served there for eleven consecutive years. The tower "
            "stands 34 metres high and its dioptric apparatus of the second order shows two white "
            "flashes every fifteen seconds, visible at 18 nautical miles. The station burns "
            "kerosene at a rate of 460 litres per month, drawn from a store built into the head "
            "keeper's cottage. A fog signal of two blasts per minute was added in 1904 after the "
            "wreck of the barque Santa Ana, which struck the reef beneath the point on 9 July 1903 "
            "with the loss of seven hands. The point lies 6 kilometres north of the fishing port of "
            "Praia Seca. A telephone line to the port was completed in 1907. Ferreira was succeeded "
            "in 1912 by his assistant, Joao Braz, who had joined the service in 1898."
        ),
        question="How far is the lighthouse visible?",
        expected_answer="18 nautical miles",
        facts=[
            "The lighthouse at Cape Verde Point was commissioned on 3 March 1901.",
            "The first keeper was Augusto Ferreira.",
            "Ferreira served for eleven consecutive years.",
            "The tower stands 34 metres high.",
            "The apparatus shows two white flashes every fifteen seconds.",
            "The light is visible at 18 nautical miles.",
            "The station burns kerosene at 460 litres per month.",
            "The fog signal gives two blasts per minute.",
            "The fog signal was added in 1904.",
            "It was added after the wreck of the barque Santa Ana.",
            "The Santa Ana struck the reef on 9 July 1903.",
            "Seven hands were lost in the wreck.",
            "The point lies 6 kilometres north of Praia Seca.",
            "A telephone line to the port was completed in 1907.",
            "Ferreira was succeeded in 1912 by Joao Braz.",
            "Joao Braz had joined the service in 1898.",
        ],
    ),
    dict(
        id="fact_med_03", register="fact", len_class="medium",
        passage=(
            "The Rio Tinto Copper Company shipped 1,240 ingots of refined copper from Huelva on 28 "
            "May 1888 aboard the sailing barque Andromache, bound for Swansea. The ingots weighed "
            "in total 112 tons and were valued at 8,400 pounds sterling. The Andromache, a vessel "
            "of 598 tons register under Captain Ruiz, expected to make the passage in eleven days "
            "but was delayed by calms off Cape St. Vincent for four days. Insurance for the cargo "
            "was written at Lloyd's at a premium of one and three-quarters percent. The company's "
            "agent in Swansea, Morris and Sons, was instructed to sell no ingot below 64 pounds per "
            "ton. Of the 1,240 ingots, 300 were stamped with the company mark of 1887 and the "
            "remainder with the mark of 1888. Two ingots were set aside at Huelva as assay samples "
            "and were not shipped."
        ),
        question="Where was the copper cargo bound for?",
        expected_answer="Swansea",
        facts=[
            "The Rio Tinto Copper Company shipped 1,240 ingots of refined copper from Huelva.",
            "The shipment date was 28 May 1888.",
            "The carrying vessel was the sailing barque Andromache.",
            "The cargo was bound for Swansea.",
            "The ingots weighed 112 tons in total.",
            "The cargo was valued at 8,400 pounds sterling.",
            "The Andromache was of 598 tons register.",
            "Her captain was Captain Ruiz.",
            "The expected passage was eleven days.",
            "She was delayed by calms off Cape St. Vincent for four days.",
            "Insurance was written at Lloyd's at a premium of one and three-quarters percent.",
            "The agent in Swansea was Morris and Sons.",
            "No ingot was to be sold below 64 pounds per ton.",
            "300 ingots were stamped with the company mark of 1887.",
            "Two ingots were set aside at Huelva as assay samples and not shipped.",
        ],
    ),
    dict(
        id="fact_med_04", register="fact", len_class="medium",
        passage=(
            "The railway accident at Kipton Crossing occurred on the evening of 18 April 1891 when "
            "the Toledo express, running twenty minutes late at an estimated 45 miles per hour, "
            "struck a stationary freight train that had been shunted onto the main line. The "
            "collision killed the express engineer, Michael O'Hara, and the fireman, and injured "
            "nine passengers, none fatally. The investigating board found that the station agent at "
            "Oberlin, James Caldwell, had failed to deliver the order fixing a meeting point at "
            "Elyria. The express consisted of a locomotive, a baggage car, two coaches, and a "
            "sleeper; 61 passengers were aboard according to the conductor's count at Cleveland. "
            "Damage to rolling stock was assessed at 14,200 dollars. The board recommended the "
            "introduction of the block signal system over the whole division and the dismissal of "
            "Caldwell, both of which were carried out by the company within the year."
        ),
        question="How fast was the Toledo express travelling when it struck the freight train?",
        expected_answer="45 miles per hour",
        facts=[
            "The accident at Kipton Crossing occurred on 18 April 1891.",
            "The Toledo express was running twenty minutes late.",
            "The express was travelling at an estimated 45 miles per hour.",
            "It struck a stationary freight train shunted onto the main line.",
            "The collision killed the express engineer Michael O'Hara.",
            "The fireman was also killed.",
            "Nine passengers were injured, none fatally.",
            "The station agent at Oberlin was James Caldwell.",
            "Caldwell failed to deliver the order fixing a meeting point at Elyria.",
            "The express consisted of a locomotive, a baggage car, two coaches, and a sleeper.",
            "61 passengers were aboard.",
            "The conductor's count was taken at Cleveland.",
            "Damage to rolling stock was assessed at 14,200 dollars.",
            "The board recommended the block signal system over the whole division.",
            "Both recommendations were carried out within the year.",
        ],
    ),
    dict(
        id="fact_med_05", register="fact", len_class="medium",
        passage=(
            "The tea clipper Fiery Star cleared Foochow on 21 June 1866 with 1,207 chests of tea "
            "for London, drawing 19 feet of water. Her owner, George Thompson of Aberdeen, had "
            "fitted her with Green's patent topsail yards that season, and she made 328 miles in "
            "her best day's run, logged on 30 June in the China Sea. She anchored at Anjer on 8 "
            "July, took on 11 tons of coal and fresh provisions, and sailed again the same evening. "
            "The passage to the Lizard took 89 days, the ship arriving on 18 September, six days "
            "behind the Serica, which carried the season's first market. The Fiery Star's cargo "
            "sold at an average of two shillings and ninepence per pound. Her crew numbered 33 "
            "hands, of whom four were apprentices; one apprentice, David Bruce, kept a log of the "
            "voyage later deposited at the Aberdeen Maritime Museum."
        ),
        question="How many chests of tea did the Fiery Star carry?",
        expected_answer="1,207 chests",
        facts=[
            "The tea clipper Fiery Star cleared Foochow on 21 June 1866.",
            "She carried 1,207 chests of tea for London.",
            "She was drawing 19 feet of water.",
            "Her owner was George Thompson of Aberdeen.",
            "She was fitted with Green's patent topsail yards that season.",
            "Her best day's run was 328 miles.",
            "The best run was logged on 30 June in the China Sea.",
            "She anchored at Anjer on 8 July.",
            "At Anjer she took on 11 tons of coal and fresh provisions.",
            "The passage to the Lizard took 89 days.",
            "She arrived on 18 September.",
            "She arrived six days behind the Serica.",
            "The cargo sold at an average of two shillings and ninepence per pound.",
            "Her crew numbered 33 hands.",
            "Four of the crew were apprentices.",
            "The apprentice David Bruce kept a log of the voyage.",
        ],
    ),
    dict(
        id="fact_med_06", register="fact", len_class="medium",
        passage=(
            "The mining company dispatch from Broken Hill, dated 15 September 1893, reported that "
            "the new shaft had reached the lode at 340 feet, eleven weeks after sinking began on 1 "
            "July. The assay of the first crosscut returned 22 ounces of silver to the ton over a "
            "width of 4 feet. The company, Silver Queen Proprietary, employed 87 men underground "
            "and 44 on the surface, under mine manager Daniel Callaghan, who had come from "
            "Kalgoorlie that April. Two winding engines of 60 indicated horsepower each had been "
            "erected on the surface, and a third was on order from Mort's Foundry of Sydney at a "
            "cost of 1,850 pounds. Water ingress at the shaft bottom was measured at 900 gallons "
            "per hour and handled by a Cornish pump. The dispatch further noted that wages had "
            "been raised by two shillings per shift following the strike of August, and that the "
            "amalgam retorts had yielded 3,100 ounces of silver for the month."
        ),
        question="At what depth did the new shaft reach the lode?",
        expected_answer="340 feet",
        facts=[
            "The dispatch from Broken Hill was dated 15 September 1893.",
            "The new shaft reached the lode at 340 feet.",
            "Sinking began on 1 July.",
            "Eleven weeks passed after sinking began.",
            "The first crosscut assayed 22 ounces of silver to the ton.",
            "The lode width was 4 feet.",
            "The company was Silver Queen Proprietary.",
            "87 men were employed underground.",
            "44 men were employed on the surface.",
            "The mine manager was Daniel Callaghan.",
            "Callaghan had come from Kalgoorlie that April.",
            "Two winding engines of 60 indicated horsepower each had been erected.",
            "A third engine was on order from Mort's Foundry of Sydney.",
            "The third engine cost 1,850 pounds.",
            "Water ingress was 900 gallons per hour.",
            "Wages were raised by two shillings per shift following the strike of August.",
            "The retorts yielded 3,100 ounces of silver for the month.",
        ],
    ),
    # ---------------- long (~1500-2000 chars) ----------------
    dict(
        id="fact_long_01", register="fact", len_class="long",
        passage=(
            "The commission appointed to inquire into the loss of the Royal Charter steam clipper "
            "held its sittings at Liverpool beginning on 6 February 1860, before Mr. Tremenheere, "
            "assisted by Captains Harris and Gray of the Board of Trade. The ship, of 2,719 tons "
            "register, built on the Clyde at Sandycroft in 1855, had left Melbourne on 26 August "
            "1859 with 388 passengers, a crew of 112, and a general cargo that included 79,000 "
            "ounces of gold in 39 boxes consigned to London banks. She was commanded by Captain "
            "John Taylor, who had sailed with the ship since her first voyage. After a passage of "
            "fifty-nine days against heavy westerly winds, she rounded the Mull of Galloway on the "
            "morning of 25 October and took two pilots aboard at Point Lynas. That evening the "
            "wind backed to the northeast and increased to hurricane force by ten o'clock. The "
            "engines, of 200 nominal horsepower, were worked to their utmost, but the ship drifted "
            "stern-first toward the coast of Anglesey. At half past one in the morning of 26 "
            "October she struck the rocks half a mile east of Moelfre. The first boat launched was "
            "stove against the ship's side; a raft constructed by the carpenter, Peter Gibson, "
            "carried the purser and twenty men to shore, and a buoy line rigged by the mate, "
            "Joseph Rodgers, saved a further nineteen. In all, 41 persons survived of the 500 "
            "aboard. The gold, with one exception, went down with the ship; a single box "
            "containing 1,640 ounces was recovered from the beach at Moelfre in November by a "
            "local man, Robert Williams. The commission found that the ship had been seaworthy "
            "and well found, that the pilots had advised standing out to sea when the wind "
            "shifted, and that the decision to anchor in Moelfre roads had been taken by Captain "
            "Taylor against that advice. They recommended the establishment of a storm signal "
            "station at Point Lynas and a lifeboat at Moelfre, both of which were provided within "
            "two years by the Marine Board."
        ),
        question="How many people survived the wreck of the Royal Charter?",
        expected_answer="41",
        facts=[
            "The commission held its sittings at Liverpool beginning on 6 February 1860.",
            "The Royal Charter was of 2,719 tons register.",
            "She was built on the Clyde at Sandycroft in 1855.",
            "She had left Melbourne on 26 August 1859.",
            "She carried 388 passengers.",
            "She carried a crew of 112.",
            "The cargo included 79,000 ounces of gold in 39 boxes.",
            "The gold was consigned to London banks.",
            "She was commanded by Captain John Taylor.",
            "The passage to the Mull of Galloway took fifty-nine days.",
            "She rounded the Mull of Galloway on the morning of 25 October.",
            "She took two pilots aboard at Point Lynas.",
            "The wind backed to the northeast and reached hurricane force by ten o'clock.",
            "Her engines were of 200 nominal horsepower.",
            "She struck the rocks half a mile east of Moelfre.",
            "The wreck occurred at half past one in the morning of 26 October.",
            "The carpenter was Peter Gibson.",
            "The carpenter's raft carried the purser and twenty men to shore.",
            "A buoy line rigged by the mate Joseph Rodgers saved nineteen more.",
            "41 persons survived of the 500 aboard.",
            "A single box containing 1,640 ounces of gold was recovered from the beach.",
            "The gold box was recovered in November.",
            "It was recovered by a local man named Robert Williams.",
            "The pilots had advised standing out to sea when the wind shifted.",
            "The decision to anchor in Moelfre roads was taken by Captain Taylor against that advice.",
            "The commission recommended a storm signal station at Point Lynas.",
            "The commission recommended a lifeboat at Moelfre.",
            "Both recommendations were provided within two years.",
        ],
    ),
    dict(
        id="fact_long_02", register="fact", len_class="long",
        passage=(
            "The correspondence of the Baltic timber firm of Hagen and Sons for March 1871 opens "
            "with a letter from Christiania, dated 2 March, in which the senior partner, Ludvig "
            "Hagen, instructs his brother Oscar, then at Riga, to purchase no more than 3,000 "
            "loads of pine deals at prices above 11 shillings per load, since advices from Hull "
            "reported a falling market. Oscar's reply of 9 March, carried by the steamer Denmark, "
            "states that he had secured 2,400 loads of first-quality Memel deals at 10 shillings "
            "and sixpence, and had declined a further 800 loads offered at 12 shillings. The "
            "firm's Hull agent, Whitfield and Company, wrote on 17 March that the arrivals of the "
            "past fortnight had been 42 ships against 29 in the same period of the previous year, "
            "and that best Memel deals had fallen to 13 shillings and fourpence, a decline of "
            "ninepence. They further reported the failure of the house of Robertson and Maine, "
            "which had held engagements in the timber trade to the amount of 18,000 pounds, an "
            "event expected to depress prices further. On 24 March Ludvig wrote to the firm's "
            "bankers, the Discount Bank of Christiania, arranging an overdraft of 6,000 pounds at "
            "five percent to meet the drafts falling due in April. The letter also records that "
            "the bark Alida had cleared Riga on 14 March with 1,100 deals consigned to Hull, "
            "insured at Lloyd's for 2,750 pounds, and that the steamer Denmark had been chartered "
            "for the return voyage at 45 pounds per day. A final note of 31 March records the "
            "engagement of a new clerk, one Nils Petersen, at a salary of 120 pounds per annum, "
            "and the purchase for the counting-house of a second-hand copying press for 4 pounds "
            "and ten shillings from the estate sale of a retiring merchant."
        ),
        question="At what price had Oscar Hagen secured the Memel deals?",
        expected_answer="10 shillings and sixpence",
        facts=[
            "The senior partner was Ludvig Hagen.",
            "Ludvig wrote from Christiania on 2 March 1871.",
            "He instructed his brother Oscar, then at Riga, to purchase no more than 3,000 loads.",
            "The price limit was 11 shillings per load.",
            "Advices from Hull reported a falling market.",
            "Oscar's reply was dated 9 March.",
            "Oscar's reply was carried by the steamer Denmark.",
            "Oscar had secured 2,400 loads of first-quality Memel deals.",
            "The price secured was 10 shillings and sixpence.",
            "He declined a further 800 loads offered at 12 shillings.",
            "The Hull agent was Whitfield and Company.",
            "Their letter was dated 17 March.",
            "Arrivals in the past fortnight had been 42 ships.",
            "The previous year's same period had 29 ships.",
            "Best Memel deals had fallen to 13 shillings and fourpence, a decline of ninepence.",
            "The house of Robertson and Maine had failed.",
            "Robertson and Maine held engagements of 18,000 pounds.",
            "Ludvig arranged an overdraft of 6,000 pounds at five percent.",
            "The overdraft was with the Discount Bank of Christiania.",
            "The overdraft was to meet drafts falling due in April.",
            "The bark Alida cleared Riga on 14 March.",
            "The Alida carried 1,100 deals consigned to Hull.",
            "The Alida's cargo was insured for 2,750 pounds.",
            "The steamer Denmark was chartered for the return voyage at 45 pounds per day.",
            "The new clerk was Nils Petersen.",
            "His salary was 120 pounds per annum.",
            "The copying press cost 4 pounds and ten shillings.",
            "It was bought from the estate sale of a retiring merchant.",
        ],
    ),
    dict(
        id="fact_long_03", register="fact", len_class="long",
        passage=(
            "The annual report of the Submarine Telegraph Company for the year ending 31 December "
            "1868 was laid before the shareholders at a general meeting held at the offices in "
            "Lothbury, London, on 11 February 1869, the chair being taken by the company's "
            "president, Sir Charles Bright. The report stated that the company now operated 1,940 "
            "nautical miles of submarine cable, of which 310 miles had been laid during the year "
            "at a cost of 118,000 pounds, comprising the new Dover to Cape Grisnez circuit of 26 "
            "miles and the extension from Cromer to Hanstholm of 284 miles. The gross receipts "
            "for the year were 147,300 pounds, an increase of 11,200 pounds over 1867, of which "
            "the Continental messages contributed 96,400 pounds at an average of four shillings "
            "and twopence per message. The total number of messages transmitted was 412,000, "
            "being a daily average of 1,127. Working expenses were stated at 62,500 pounds, "
            "leaving a net profit of 84,800 pounds, from which the directors recommended a "
            "dividend of seven percent, absorbing 59,000 pounds, and the carrying of 25,800 "
            "pounds to the reserve fund, which would then stand at 100,000 pounds. The engineer "
            "to the company, Mr. Latimer Clark, reported that the failures of the year had been "
            "confined to two faults in the Dover cable, the first on 14 April caused by a ship's "
            "anchor at the Goodwins, repaired in nineteen hours, and the second on 3 August near "
            "the French shore, repaired in two days by the company's steamer Investigator under "
            "Captain Wood. The staff at the date of the report numbered 214 persons, of whom 61 "
            "were clerks employed at the central station in Lothbury, working in three shifts of "
            "eight hours. The meeting closed with a vote of thanks to the chairman, carried "
            "unanimously."
        ),
        question="What dividend did the directors recommend?",
        expected_answer="seven percent",
        facts=[
            "The report covered the year ending 31 December 1868.",
            "The general meeting was held on 11 February 1869.",
            "It was held at the offices in Lothbury, London.",
            "The president was Sir Charles Bright.",
            "The company operated 1,940 nautical miles of submarine cable.",
            "310 miles had been laid during the year.",
            "The new cable had cost 118,000 pounds.",
            "The Dover to Cape Grisnez circuit was 26 miles.",
            "The Cromer to Hanstholm extension was 284 miles.",
            "Gross receipts for the year were 147,300 pounds.",
            "This was an increase of 11,200 pounds over 1867.",
            "Continental messages contributed 96,400 pounds.",
            "The average message price was four shillings and twopence.",
            "The total number of messages transmitted was 412,000.",
            "The daily average was 1,127 messages.",
            "Working expenses were 62,500 pounds.",
            "Net profit was 84,800 pounds.",
            "The directors recommended a dividend of seven percent.",
            "The dividend absorbed 59,000 pounds.",
            "25,800 pounds was carried to the reserve fund.",
            "The reserve fund would then stand at 100,000 pounds.",
            "The engineer was Mr. Latimer Clark.",
            "There were two faults in the Dover cable during the year.",
            "The first fault was on 14 April, caused by a ship's anchor at the Goodwins.",
            "The first fault was repaired in nineteen hours.",
            "The second fault was on 3 August, near the French shore.",
            "The second fault was repaired in two days.",
            "The repairing steamer was the Investigator under Captain Wood.",
            "The staff numbered 214 persons.",
            "61 clerks worked at the central station in Lothbury.",
            "The clerks worked in three shifts of eight hours.",
        ],
    ),
    dict(
        id="fact_long_04", register="fact", len_class="long",
        passage=(
            "Field dispatch from the survey of the Bengal Nagpur railway extension, written at "
            "Chanda on 29 November 1889 by the assistant engineer, Mr. Reginald Shaw, to the "
            "chief engineer at Calcutta. The dispatch reports that the survey party, numbering 34 "
            "persons including 12 chainmen and 6 sappers of the Madras Sappers, had completed the "
            "location survey of the 92-mile section from Nagbhid to Chanda on 22 November, after "
            "fourteen weeks in the field. The line as located follows the Wyne Ganga valley for "
            "41 miles, crosses that river by a bridge of seven spans of 150 feet at Karanja, and "
            "then takes the northern flank of the hills. Shaw estimates earthwork at 480,000 "
            "cubic yards for the whole section, of which 62,000 cubic yards are rock excavation "
            "in the eleven miles beyond Ballarshah. Three alignments were examined for the "
            "Karanja crossing; the adopted line is 1.7 miles longer than the southern route but "
            "avoids 90,000 cubic yards of rock and two well sinkings. The country between "
            "Ballarshah and Chanda is reported healthy except in the months of July and August, "
            "when fever carried off four chainmen this season. The party's expenses to date were "
            "19,400 rupees against a sanctioned grant of 30,000 rupees. Shaw requests an "
            "additional 8,000 rupees for the next season, a theodolite to replace the one damaged "
            "at the Wyne Ganga crossing on 3 September, and the loan of a second survey elephant "
            "from the Chanda division, the party's own elephant having gone lame after the "
            "monsoon. He closes by noting that the levelling of the first 40 miles has been "
            "checked twice and agrees within 0.15 feet."
        ),
        question="How long is the section surveyed, in miles?",
        expected_answer="92 miles",
        facts=[
            "The dispatch was written at Chanda on 29 November 1889.",
            "It was written by the assistant engineer Mr. Reginald Shaw.",
            "It was addressed to the chief engineer at Calcutta.",
            "The survey party numbered 34 persons.",
            "The party included 12 chainmen.",
            "The party included 6 sappers of the Madras Sappers.",
            "The section surveyed runs from Nagbhid to Chanda.",
            "The section is 92 miles long.",
            "The location survey was completed on 22 November.",
            "The survey took fourteen weeks in the field.",
            "The line follows the Wyne Ganga valley for 41 miles.",
            "The Wyne Ganga is crossed by a bridge of seven spans of 150 feet at Karanja.",
            "Shaw estimates earthwork at 480,000 cubic yards for the whole section.",
            "62,000 cubic yards are rock excavation.",
            "The rock excavation lies in the eleven miles beyond Ballarshah.",
            "Three alignments were examined for the Karanja crossing.",
            "The adopted line is 1.7 miles longer than the southern route.",
            "It avoids 90,000 cubic yards of rock and two well sinkings.",
            "Fever carried off four chainmen this season, in July and August.",
            "Expenses to date were 19,400 rupees.",
            "The sanctioned grant was 30,000 rupees.",
            "Shaw requests an additional 8,000 rupees for the next season.",
            "He requests a theodolite to replace one damaged at the Wyne Ganga crossing.",
            "The theodolite was damaged on 3 September.",
            "He requests the loan of a second survey elephant from the Chanda division.",
            "The party's own elephant went lame after the monsoon.",
            "The levelling of the first 40 miles has been checked twice.",
            "The two levellings agree within 0.15 feet.",
        ],
    ),
    dict(
        id="fact_long_05", register="fact", len_class="long",
        passage=(
            "The consulate report on the wool trade of the La Plata region, drawn up at Buenos "
            "Aires by Her Majesty's Consul, Mr. Edmund Monson, and forwarded to the Foreign "
            "Office on 20 April 1875, records that the clip of the season just closed was "
            "estimated at 68,000,000 pounds weight, against 61,000,000 pounds in the previous "
            "season, the increase being chiefly assigned to the extension of fencing on the "
            "estancias of the southern camps. Of this clip, some 26,000 bales were shipped to "
            "Havre, 19,000 bales to Antwerp, and 11,500 bales to Great Britain, the remainder "
            "being worked up in the local factories, of which fourteen were reported in "
            "operation against nine a year before. The average price realized at the sales was "
            "elevenpence per pound greasy, a fall of twopence farthing on the year. The consul "
            "notes that the S.A. Bienvenida sheep, introduced by the Sociedad Rural in 1868, now "
            "numbered, by the best estimates, 3,000,000 head, and that their wool brought a "
            "premium of one penny per pound over the common criollo clip. Freight to London was "
            "taken at five eighths of a penny per pound, and to Havre at a halfpenny farthing; "
            "insurance on London account ran at thirty-five shillings percent. The report "
            "concludes with the observation that the drainage of the flooding lands near "
            "Dolores, begun in 1873 under the engineer Mr. James Baine, had reclaimed some "
            "400 square miles of pasture, and that the railway from Tandil to the port, opened "
            "on 1 February of the present year, had already reduced cartage charges by two "
            "shillings per bale."
        ),
        question="What was the estimated weight of the season's wool clip?",
        expected_answer="68,000,000 pounds",
        facts=[
            "The report was drawn up at Buenos Aires.",
            "It was drawn up by Her Majesty's Consul Mr. Edmund Monson.",
            "It was forwarded to the Foreign Office on 20 April 1875.",
            "The clip of the season was estimated at 68,000,000 pounds weight.",
            "The previous season's clip was 61,000,000 pounds.",
            "The increase was chiefly assigned to the extension of fencing on the southern camps.",
            "26,000 bales were shipped to Havre.",
            "19,000 bales were shipped to Antwerp.",
            "11,500 bales were shipped to Great Britain.",
            "Fourteen local factories were in operation.",
            "A year before there had been nine factories.",
            "The average price at the sales was elevenpence per pound greasy.",
            "This was a fall of twopence farthing on the year.",
            "The S.A. Bienvenida sheep was introduced by the Sociedad Rural in 1868.",
            "The Bienvenida sheep numbered about 3,000,000 head.",
            "Their wool brought a premium of one penny per pound over the criollo clip.",
            "Freight to London was five eighths of a penny per pound.",
            "Freight to Havre was a halfpenny farthing per pound.",
            "Insurance on London account ran at thirty-five shillings percent.",
            "The drainage of the lands near Dolores was begun in 1873.",
            "The drainage engineer was Mr. James Baine.",
            "The drainage had reclaimed about 400 square miles of pasture.",
            "The railway from Tandil to the port opened on 1 February of the present year.",
            "The railway had reduced cartage charges by two shillings per bale.",
        ],
    ),
    dict(
        id="fact_long_06", register="fact", len_class="long",
        passage=(
            "Log of the whaler Diana of Hull, kept by her surgeon, Dr. Charles Edward Smith, "
            "during the Davis Strait fishery of 1866. The Diana, a brig-rigged steam whaler of "
            "350 tons register, commanded by Captain John Gravill, left Hull on 12 February with "
            "a crew of 49 hands, of whom 8 were green hands shipped at two pounds ten per month, "
            "the seasoned men drawing three pounds five. She crossed the Arctic circle on 3 "
            "March in longitude 22 degrees west and made the ice edge off Disko on 19 March. "
            "The first whale was struck on 2 April off the Women's Islands and yielded 12 tons "
            "of oil and 9 hundredweight of bone; a second, taken on 17 April, yielded 15 tons. "
            "By the end of May the catch stood at four whales and 1,400 seals. On 11 June the "
            "ship was nipped in the ice in Melville Bay and lost her rudder; a temporary one was "
            "fashioned from spare spars by the carpenter within three days. The ship was then "
            "beset and drifted with the pack for 61 days, during which the surgeon recorded the "
            "first case of scurvy on 8 July and, by the middle of August, 22 men on the sick "
            "list. Lime juice having run out, the crew ate sorrel gathered on the Greenland "
            "coast when the ship touched land on 21 August. Three men died in the last week of "
            "August and were buried on shore. The Diana worked free of the ice on 25 September "
            "and reached Hull on 11 October with 41 tons of oil and 3 tons of bone, a paying "
            "voyage only by the seal catch. The owners, Samuel and Charles Gourlay, declared "
            "the gross earnings at 3,860 pounds and paid each seasoned hand 19 pounds twelve "
            "shillings for the voyage."
        ),
        question="How many days did the Diana drift with the ice pack?",
        expected_answer="61 days",
        facts=[
            "The log was kept by the surgeon Dr. Charles Edward Smith.",
            "The fishery was the Davis Strait fishery of 1866.",
            "The Diana was a brig-rigged steam whaler of 350 tons register.",
            "Her captain was John Gravill.",
            "She left Hull on 12 February.",
            "The crew numbered 49 hands.",
            "8 of the crew were green hands.",
            "Green hands were shipped at two pounds ten per month.",
            "Seasoned men drew three pounds five per month.",
            "She crossed the Arctic circle on 3 March.",
            "The crossing was in longitude 22 degrees west.",
            "She made the ice edge off Disko on 19 March.",
            "The first whale was struck on 2 April off the Women's Islands.",
            "The first whale yielded 12 tons of oil and 9 hundredweight of bone.",
            "A second whale was taken on 17 April.",
            "The second whale yielded 15 tons of oil.",
            "By the end of May the catch was four whales and 1,400 seals.",
            "The ship was nipped in the ice in Melville Bay on 11 June.",
            "She lost her rudder in the nipping.",
            "A temporary rudder was fashioned within three days.",
            "The ship drifted with the pack for 61 days.",
            "The first case of scurvy was recorded on 8 July.",
            "By mid-August 22 men were on the sick list.",
            "The crew ate sorrel gathered on the Greenland coast.",
            "The ship touched land on 21 August.",
            "Three men died in the last week of August.",
            "The Diana worked free of the ice on 25 September.",
            "She reached Hull on 11 October.",
            "She returned with 41 tons of oil and 3 tons of bone.",
            "The owners were Samuel and Charles Gourlay.",
            "Gross earnings were declared at 3,860 pounds.",
            "Each seasoned hand was paid 19 pounds twelve shillings.",
        ],
    ),
]


def main():
    for task in TASKS:
        path = os.path.join(HERE, task["id"] + ".json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(task, fh, sort_keys=True, indent=2, ensure_ascii=False)
            fh.write("\n")
    print(f"{len(TASKS)} fact-shaped tasks written")


if __name__ == "__main__":
    main()
