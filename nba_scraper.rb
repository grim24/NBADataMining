require 'rubygems'
require 'nokogiri'
require 'open-uri'
require 'rest-client'

def parse_game(score_page)
  #parse score
  score = score_page.search('div#all_line_score')
  score.xpath('//comment()').each { |comment| comment.replace(comment.text) }
  score_table =  score.css("table")
  scores = score_table.css("td").css("strong")
  teams = score_table.css("td").css("a")
  home_score = scores[1].text
  away_score = scores[0].text
  home_team = teams[1].text
  away_team = teams[0].text

  home_result = "n/a"
  away_result = "n/a"
  if home_score.to_i > away_score.to_i
    home_result = "win"
    away_result = "loss"
  else
    home_result = "loss"
    away_result = "win"
  end

  #parse 4 factors
  factors = score_page.search('div#all_four_factors')
  factors.xpath('//comment()').each { |comment| comment.replace(comment.text) }
  factors_table = factors.css("table")
  efg_pct = factors_table.css("td[data-stat='efg_pct']")
  tov_pct = factors_table.css("td[data-stat='tov_pct']")
  orb_pct = factors_table.css("td[data-stat='orb_pct']")
  ft_rate = factors_table.css("td[data-stat='ft_rate']")

  home_eft_pct = efg_pct[1].text
  home_tov_pct = tov_pct[1].text
  home_orb_pct = orb_pct[1].text
  home_ft_rate = ft_rate[1].text
  away_eft_pct = efg_pct[0].text
  away_tov_pct = tov_pct[0].text
  away_orb_pct = orb_pct[0].text
  away_ft_rate = ft_rate[0].text


  puts away_team + ", " + away_score.to_s + ", " + away_eft_pct.to_s + ", " + away_tov_pct.to_s + ", " + away_orb_pct.to_s + ", " + away_ft_rate.to_s + ", " + away_result
  puts home_team + ", " + home_score.to_s + "," + home_eft_pct.to_s + "," + home_tov_pct.to_s + "," + home_orb_pct.to_s + "," + home_ft_rate.to_s + ", " + home_result

end

puts "Team, score,  Efg_pct, Turnover_PCT, orb_pct, ft_rate, class"

(1..12).each do |month|
  (1..31).each do |day|

    address = "http://www.basketball-reference.com/boxscores/index.fcgi?month=" + month.to_s + "&day=" + day.to_s + "&year=" + 2016.to_s

    page = Nokogiri::HTML(RestClient.get(address))   
    teams = page.css("table").css(".teams")

    links = []
    refs = teams.css("td.gamelink").css("a")
    refs.each do |ref|
      links << (ref['href'])
    end


    prefix = "http://www.basketball-reference.com/"
    links.each do |link|
      parse_game(Nokogiri::HTML(RestClient.get(prefix + link)))
    end

  end
end


