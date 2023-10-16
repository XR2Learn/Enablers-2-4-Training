Remove-Item datasets/* -r
Remove-Item outputs/* -r

Write-Output "--------------------"
Write-Output "Pre-processing-audio"
Write-Output "--------------------"
docker compose run --rm pre-processing-audio


Write-Output "--------------------"
Write-Output "Handcrafted-features-generation-audio"
Write-Output "--------------------"
docker compose run --rm handcrafted-features-generation-audio


Write-Output "--------------------"
Write-Output "SSL-training-audio"
Write-Output "--------------------"
docker compose run --rm ssl-audio

Write-Output "--------------------"
Write-Output "SSL-features-extraction-audio"
Write-Output "--------------------"
docker compose run --rm ssl-features-generation-audio

Write-Output "--------------------"
Write-Output "Supervised-training-audio"
Write-Output "--------------------"
docker compose run --rm ed-training-audio

Write-Output "--------------------"


