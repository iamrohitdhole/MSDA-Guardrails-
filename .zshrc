i# Set Java Environment Variables
export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)
export PATH=$JAVA_HOME/bin:$PATH

# Set Hadoop Environment Variables
export HADOOP_HOME=/usr/local/hadoop
export PATH="$HADOOP_HOME/bin:$PATH"
export HDFS_NAMENODE_USER=artemis
export HDFS_DATANODE_USER=artemis
export HDFS_SECONDARYNAMENODE_USER=artemis

# Set Hive Environment Variables
export HIVE_HOME=/usr/local/Cellar/hive/3.1.3/libexec
export HIVE_CONF_DIR=$HIVE_HOME/conf
export PATH=$HIVE_HOME/bin:$PATH
export HIVE_CLASSPATH=$HIVE_HOME/lib/hive-exec-3.1.3.jar



# The following lines have been added by Docker Desktop to enable Docker CLI completions.
fpath=(/Users/artemis/.docker/completions $fpath)
autoload -Uz compinit
compinit
# End of Docker CLI completions

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/Caskroom/miniconda/base/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
        . "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/Caskroom/miniconda/base/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export JAVA_HOME=$(/usr/libexec/java_home -v 11)
export PATH=$JAVA_HOME/bin:$PATH
