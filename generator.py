import runner

runMachine=runner.Runner()
finalSize=int(input("Final Size: "))
final=""
finalWithoutPunct=""
prompt=input(f"Enter a {runMachine.contextSize}-word sentence:\n").strip()
final=prompt
preppedFinal=runMachine.prep(final)
while len(preppedFinal.split())<=finalSize:
    finalBatchWords=preppedFinal.split()[len(preppedFinal.split())-runMachine.contextSize:]
    finalBatch=""
    for word in finalBatchWords:
        finalBatch+=word
        finalBatch+=" "
    result=runMachine.run(finalBatch)
    final+=" "
    final+=result
    preppedFinal=runMachine.prep(final)
print(final)