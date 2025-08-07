    #!/bin/bash
    # Install Amazon Corretto 17
    sudo yum install java-17-openjdk-devel

    # Set Java 17 as the default
    sudo update-alternatives --set java /usr/lib/jvm/java-17-amazon-corretto.x86_64/bin/java
    sudo update-alternatives --set javac /usr/lib/jvm/java-17-amazon-corretto.x86_64/bin/javac

    # Optional: Configure Spark to use Java 17
    # For example, in spark-defaults.conf
    # echo "spark.emr-serverless.driverEnv.JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto.x86_64/" | sudo tee -a /etc/spark/conf/spark-defaults.conf
    # echo "spark.executorEnv.JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto.x86_64/" | sudo tee -a /etc/spark/conf/spark-defaults.conf