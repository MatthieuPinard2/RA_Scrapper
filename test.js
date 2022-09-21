// https://ra.co/events/uk/london?week=2022-09-28

root = this.document.body.firstChild.getElementsByTagName("header")[0].nextSibling.getElementsByTagName("section")[1].getElementsByTagName("div")[0].getElementsByTagName("div")[0].getElementsByTagName("div")[1].getElementsByTagName("ul")[0].getElementsByTagName("li")[0].getElementsByTagName("div")[0]

dateStr = root.getElementsByTagName("div")[0].textContent

parties = root.getElementsByTagName("h3")

var eventsURL = []
for(var i = 1; i < parties.length; i++) {
    eventsURL.push("https://ra.co" + parties[i].getElementsByTagName("a")[0].getAttribute("href"))
}
