#!/bin/bash

mvn package -Dmaven.test.skip
cp ./target/spark-mllib_2.12-3.1.0-SNAPSHOT.jar ../assembly/target/scala-2.12/jars/spark-mllib_2.12-3.1.0-SNAPSHOT.jar
