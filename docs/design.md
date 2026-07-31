# How the Home Health Monitor should be completed

This document explains the important product and medical decisions in simple
language. The software foundation is implemented, but a trustworthy health
model cannot be invented without real examples and correct answers.

## The main idea

The sensors tell us what happened to a person's temperature, heart rate and
movement during the last day. A model can compare those patterns with patterns
from previously collected examples.

For example, it may learn that a combination of increasing body temperature,
an unusual resting heart rate and reduced movement was often followed by a
particular illness label in the training study. It cannot learn this from the
four sensor names alone. It needs many historical examples with the correct
outcome attached.

## Questions that must be answered first

### What does “sick” mean?

Someone must provide a precise answer. Possible definitions include a fever
confirmed with a trusted thermometer or an illness confirmed by a clinician.
“The person did not feel well” may be useful, but it is a different and less
consistent target.

### How far into the future are we predicting?

Choose one period, such as the next 24 hours. A model predicting the next six
hours is a different model from one predicting the next three days.

### What exactly do the sensors measure?

We need the sensor models, where they are worn or installed, their units, how
often they report and what they send when disconnected. Skin temperature is not
the same as core body temperature. A binary motion sensor also cannot explain
whether a high heart rate came from exercise, stress or illness.

### Who will use it?

Age, medication, known health conditions and normal fitness can change the
meaning of a heart-rate or temperature reading. The intended group must be
defined, and the training data must represent that group.

## Recommended data flow

1. Sensors collect readings throughout the day.
2. A collector keeps the detailed readings locally.
3. The collector creates roughly one combined reading per minute.
4. It sends the latest 24 hours to this service.
5. The service validates and summarises those readings.
6. The model returns current and future risk.

One reading per minute produces about 1,440 readings per person per day and is a
reasonable starting point for this device. Missing periods should stay missing.
Inventing readings by copying the last known value can mislead the model.

## Why the first model is simple

The first model is logistic regression. In ordinary language, it gives each
summary a learned importance, combines them and converts the answer to a value
between zero and one.

This is a good starting point because it is:

- tiny enough for the 512 MB Pi;
- fast;
- easy to inspect;
- less likely to memorise a small pilot study;
- suitable for probability checks.

A random forest or a small boosted-tree model should also be tested later. We
should use one only if tests on completely different people show that it is
meaningfully better. A random forest file may be small, but loading the full
scikit-learn environment on the Pi wastes memory that the exported logistic
model does not need.

## What the model receives

The raw day's readings are turned into summaries for the most recent 1, 6 and
24 hours. These include:

- average, lowest and highest values;
- how much each value changes;
- whether it is rising or falling;
- the latest value;
- the percentage of readings with movement;
- changes between moving and not moving;
- the average difference between body and room temperature;
- the amount of history and the largest missing interval.

This turns thousands of readings into a small fixed list of numbers, making the
prediction cheap to run.

## How to test whether it works

Readings from one person must not appear on both sides of a test. Otherwise, the
model may recognise that person's normal pattern instead of learning a pattern
that works for new people.

The test report should answer:

- How many genuinely higher-risk cases did it find?
- How many did it miss?
- How many lower-risk people received a false warning?
- When it reports 70% risk, does the event happen approximately 70% of the time?
- Does it work similarly for the important groups in the intended population?
- What happens when readings are missing or a sensor behaves differently?

Overall “accuracy” is not enough. If illness is rare, a system that always says
“lower risk” can appear accurate while being useless.

## Safety

Model results must not be the only emergency system. If the product needs an
alarm for a dangerous measurement, clinicians must define that separate rule.
It must continue working even when the AI model is missing or unavailable.

The accepted ranges in the current code only reject clearly broken input. They
are not medical emergency limits.

Every response says that it is a screening result, not a diagnosis. The product
interface should also tell users what to do when they feel seriously unwell,
regardless of the model result.

## Privacy and device security

Health readings are sensitive. Use meaningless private IDs rather than names,
do not log complete requests, keep only data that is needed and define who can
access or delete it.

The tiny HTTP service does not provide login or encryption. On the Pi it listens
only on the local device by default. If another machine must call it, place an
authenticated HTTPS gateway in front of it.

## What is still needed from the project owner

- The exact sickness/current-risk definitions
- The future prediction period
- Sensor details and sampling frequency
- The intended users
- A labelled dataset from multiple people
- A clinician-reviewed emergency policy, if alerts are required
- The acceptable balance between missed cases and false warnings
- A privacy, retention and access policy

Once those items and the data are available, the training code can create the
first real artifact. The resulting model must pass the tests above before it is
placed on a Pi for anything beyond a controlled pilot.
