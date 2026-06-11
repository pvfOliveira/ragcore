# Grid-Scale Energy Storage: A Reference

A working reference on stationary energy storage for electric grids. This
document is the committed eval corpus for ragcore's golden dataset
(`ragcore/eval/golden/v1.jsonl`): every golden item cites a section below.

## Overview

Grid-scale batteries store solar and wind energy so that electricity generated
at midday or on windy nights can be delivered when customers actually need it.
Grid storage is needed to balance intermittent renewable supply with demand:
photovoltaic output collapses at sunset and wind output swings with weather,
while demand follows its own daily and seasonal rhythm. A storage fleet absorbs
surplus generation, releases it during shortfalls, and in doing so reduces
curtailment of renewable plants and the need to run fossil "peaker" units.

## Technologies

The principal grid storage technologies in commercial service are:

- lithium-ion batteries
- pumped-storage hydropower
- vanadium redox flow batteries
- compressed-air energy storage (CAES), which holds pressurized air in
  underground caverns
- sodium-sulfur (NaS) batteries, operated at high temperature
- molten-salt thermal storage, paired with concentrating solar plants to
  retain heat for release over several hours
- flywheels

Each occupies a different niche of discharge duration, siting constraints, and
cost, several of which are detailed in the sections that follow.

## Lithium-Ion Batteries

Lithium-ion is the dominant technology for new grid storage deployments.
Grid-scale lithium-ion systems achieve a round-trip efficiency of 85 to 92
percent and are typically built for discharge durations of one to four hours
at rated power. Cells lose usable capacity gradually with cycling ("capacity
fade"), so projects are sized with margin and augmented over their service
life. The main safety hazard is thermal runaway, which station designs
mitigate with battery management systems, compartmentalization, and active
cooling.

## Pumped-Storage Hydropower

Pumped-storage hydropower is the oldest and largest grid storage technology:
it accounts for roughly 90 percent of installed grid storage capacity
worldwide. A plant pumps water from a lower reservoir to an upper one when
electricity is cheap and runs the water back down through turbines when it is
scarce. Round-trip efficiency is 70 to 80 percent. Plants require two
reservoirs at substantially different elevations, which limits siting, but
they routinely provide discharge durations of six to twenty hours — far longer
than typical battery installations.

## Flow Batteries

Vanadium redox flow batteries (VRFBs) store energy in liquid electrolyte held
in external tanks and convert it in a separate cell stack. The defining
property of a VRFB is that its energy capacity is decoupled from its power
rating: energy scales with the size of the electrolyte tanks, while power
scales with the size of the cell stack, so the two can be sized independently.
The vanadium electrolyte does not degrade significantly over more than twenty
years of cycling. The trade-offs are low energy density and a larger physical
footprint than lithium-ion.

## Grid Services

Storage earns revenue by stacking several distinct grid services:

- **Frequency regulation.** Batteries respond to grid-frequency deviations
  within milliseconds, far faster than thermal plants.
- **Peak shaving.** Discharging during the hours of highest demand to reduce
  stress on the network and defer upgrades to wires and substations.
- **Energy arbitrage.** Charging when wholesale prices are low and discharging
  when they are high.
- **Black start.** Energizing a section of a dead grid so that larger plants
  can restart after a blackout.

## Economics

The economics of grid storage turned during the last decade. Lithium-ion
battery pack prices fell from about $780 per kilowatt-hour in 2013 to roughly
$140 per kilowatt-hour in 2023. Project comparisons use levelized cost of
storage (LCOS): the all-in cost per megawatt-hour discharged over a system's
life, which folds in capital cost, round-trip losses, degradation, and
financing.

## Limitations and Misconceptions

- Storage does not generate energy; it shifts energy in time, absorbing
  surplus and releasing it later, and every cycle loses some energy to
  round-trip inefficiency.
- Flywheels are not suited for long-duration storage; they serve short bursts
  of seconds to minutes, chiefly for frequency regulation.
- Today's battery fleets cannot bridge multi-day wind-and-solar droughts
  (Dunkelflaute); that gap is the target of long-duration storage research
  rather than something current lithium-ion plants solve.
